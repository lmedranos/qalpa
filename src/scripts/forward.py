import os
import time
from pathlib import Path
import shutil
import subprocess
from hydra import main as hydra_main
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.data_manage import DataManager
from src.utils.trajectory import Trajectory, read_targets
from src.utils.al_config import al_paths, get_al_cfg, resolve_blocks, resolve_start, validate_al_cfg
from src.utils.candidates import candidate_name, write_candidates_meta


# Allow experiment_config.yaml to derive values such as num_context_features from list lengths
OmegaConf.register_new_resolver("len", len, replace=True)

# Resolve the repository root based on this file's location so that paths in
# global_config.yaml work regardless of the current working directory.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
OmegaConf.register_new_resolver("repo_root", lambda: _REPO_ROOT, replace=True)

# Capture the original working directory before Hydra changes it
_ORIGINAL_CWD = os.getcwd()


def n_nodes_fallback(cfg: DictConfig, n: int, target: list) -> list:
    """Prepend --num-nodes from EQdiffParam.n_nodes when the target has none (unset = sampled atom counts)."""
    if "--num-nodes" in target:
        return target
    n_nodes_cfg = cfg.EQdiffParam.get("n_nodes")
    if isinstance(n_nodes_cfg, (list, ListConfig)):
        n_nodes_cfg = n_nodes_cfg[n] if len(n_nodes_cfg) > 1 else n_nodes_cfg[0]
    if n_nodes_cfg is not None:
        return ["--num-nodes", str(int(n_nodes_cfg)), *target]
    return target


def sync_mlflow(cfg: DictConfig) -> None:
    """Import the TensorBoard logs of all iterations into MLflow (optional).

    Runs src/scripts/sync_mlflow.py with the separate MLflow environment
    (mlflow.python) so the training env stays untouched. Failures only warn.
    """
    mlcfg = cfg.get("mlflow")
    if not mlcfg or not mlcfg.get("enabled", False):
        return
    cmd = [
        str(mlcfg.python),
        str(Path(_REPO_ROOT) / "src" / "scripts" / "sync_mlflow.py"),
        "--model-root", str(cfg.EQdiffParam.model_root),
        "--tracking-uri", str(mlcfg.tracking_uri),
        "--experiment", str(cfg.experiment_name),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD)
    except (subprocess.CalledProcessError, OSError) as err:
        print(f"WARNING: MLflow sync failed ({err}); training results are unaffected.")


@hydra_main(version_base=None, config_path="../../configs", config_name="experiment_config")
def main(cfg: DictConfig):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    Configuration is provided by Hydra/OmegaConf (experiment_config.yaml and its imports).
    '''
    # Setup environment for GNN
    gpus_per_task = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    # Create the intial hdf5 data directory if it doesn't exist (experiment_config.yaml -> data.exp_data_path)
    dm = DataManager(cfg.base_dataset_path, cfg.data.exp_data_path)
    dm.hdf5_appending_flag = bool(cfg.hdf5.appending_flag)
    dm.hdf5_iter_flag = bool(cfg.hdf5.hdf5_iter_flag)
    dm.extra_properties_flag = bool(cfg.hdf5.extra_properties_flag)
    dm.initialize_data_directory()

    # Create exterapolating trajectory for properties and scores (experiment_config.yaml -> query.*)
    prop_names = list(cfg.query.props)
    from src.pipeline_blocks.score_screen import trajectory_score_names
    score_names = trajectory_score_names(cfg)   # none if query.score_reference: initial_mean
    my_trajectory = Trajectory(cfg.query.trajectory_csv, prop_names=prop_names, score_names=score_names,
                               use_num_nodes=bool(cfg.query.get("fix_num_nodes", True)))

    # Optional active-learning target planning (experiment_config.yaml -> al.*, defaults in src/configs/al.yaml)
    al = get_al_cfg(cfg)
    al_on = bool(al["enabled"])
    validate_al_cfg(cfg, al, my_trajectory)

    # Current active-learning iteration and loop settings (experiment_config.yaml -> loop.* / EQdiffParam.*)
    n = cfg.loop.current_iter
    
    if n > cfg.loop.num_iterations:
        raise ValueError(f"Current iteration {n} exceeds the total number of iterations {cfg.loop.num_iterations}.")

    # Pipeline blocks and optional resume point (eqmgen_paths.yaml -> pipeline.blocks / pipeline.start_at)
    pipeline_blocks = resolve_blocks(cfg, al)
    start_at = resolve_start(cfg, al, pipeline_blocks)
    if start_at in pipeline_blocks:
        active_blocks = set(pipeline_blocks[pipeline_blocks.index(start_at):])
    else:
        active_blocks = set(pipeline_blocks)

    print(f"\n Iteration {n}")

    # Create directories for the current iteration
    dm.create_iteration_directories(n)
    
    # Determine checkpoint to load from
    pervious_model_path = Path(cfg.EQdiffParam.model_root) / f"model_iter_{n - 1}"
    pervious_best_ckpt = pervious_model_path / "best_mol_stab.ckpt"
    pervious_last_ckpt = pervious_model_path / "run0" / "last.ckpt"
    if n > 0:
        load_ckpt = pervious_best_ckpt if pervious_best_ckpt.exists() else pervious_last_ckpt
    elif cfg.EQdiffParam.load_ckpt_from_pretrained not in (None, "None", ""):
        load_ckpt = cfg.EQdiffParam.load_ckpt_from_pretrained
    else:
        load_ckpt = None
    # Prepare current iteration's model directory
    itr_model_path = Path(cfg.EQdiffParam.model_root) / f"model_iter_{n}"
    itr_model_path.mkdir(parents=True, exist_ok=True)
    iter_best_ckpt = itr_model_path  / "best_mol_stab.ckpt"
    iter_last_ckpt = itr_model_path / "run0" / "last.ckpt"
    save_ckpt = iter_best_ckpt if iter_best_ckpt.exists() else iter_last_ckpt  

    # Step 1: Run data split
    if 'split' in active_blocks:
        print(f" Running Data split, concatenation is: {cfg.EQdiffParam.concat_datasets} ")
        if cfg.EQdiffParam.concat_datasets:
            # dataset_0 <- base HDF5, dataset_n <- molecules labelled in iteration n-1
            # (per-iteration HDF5, written only when hdf5.hdf5_iter_flag is True)
            if n > 0 and not cfg.hdf5.hdf5_iter_flag:
                raise ValueError("EQdiffParam.concat_datasets requires hdf5.hdf5_iter_flag: True")
            source = dm.input_hdf5 if n == 0 else dm.data_directory  / f"iter_{n - 1}" / "hdf5" / "output.hdf5"
            source_root = (Path(cfg.EQdiffParam.dataset_root) / f"dataset_{n}")
            if not Path(source).is_file():
                raise FileNotFoundError(f"Source HDF5 for dataset_{n} not found: {source}. Did the backward step of iteration {n - 1} finish?")
        else:
            source = dm.output_hdf5 if cfg.hdf5.appending_flag else dm.input_hdf5
            source_root = Path(cfg.EQdiffParam.dataset_root)
        if not (source_root / "raw" / "train_data.pickle").is_file():
            source_root.mkdir(parents=True, exist_ok=True)
            print(f"  Running Data split, source: {source}, target: {source_root}")
            cmd = [
                "srun",
                "--ntasks=1",
                f"--cpus-per-task={cpus_per_task}",
                f"--gres=gpu:{gpus_per_task}",
                "--exclusive",
                "--export=ALL",
                "python",
                str(Path(cfg.EQdiff.split)),
                "--file-path",
                str(source),
                "--out-path",
                str(source_root),
                "--seed",
                str(cfg.EQdiffParam.get("split_seed", 42)),
            ]
            subprocess.run(
                cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy()
            )

    # step 2: Diffusion model training
    if 'train' in active_blocks:
        print("  Running Diffusion model training ")
        if cfg.EQdiffParam.concat_datasets:
            # Training on datasets 0..n requires dataset_n (from the split block)
            expected = Path(cfg.EQdiffParam.dataset_root) / f"dataset_{n}" / "raw" / "train_data.pickle"
            if not expected.is_file():
                raise FileNotFoundError(
                    f"{expected} not found. Run the 'split' block for iteration {n} "
                    "(pipeline.start_at: split) before training."
                )
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(cfg.EQdiff.train),
            "--conf", str(Path(cfg.code_paths.configs) / "eqdiff_model.yaml"),
            "--save-dir", str(itr_model_path),
            "--dataset", str(cfg.EQdiffParam.dataset),
            "--dataset-root", str(Path(cfg.EQdiffParam.dataset_root)),
            "--gpus", str(gpus_per_task),
            "--batch-size", str(cfg.EQdiffParam.batch_size_tr),
            "--num-epochs", str(cfg.EQdiffParam.num_epochs),
            "--test-interval", str(cfg.EQdiffParam.test_interval),
            ]
        if load_ckpt is not None:
            cmd += ["--load-ckpt-from-pretrained", str(load_ckpt)]
        # Property conditioning (EQdiffParam.context_mapping / properties_list); without these
        # run_train.py falls back to its defaults (context_mapping=False, properties_list=[]).
        if cfg.EQdiffParam.get("context_mapping", False):
            props = list(cfg.EQdiffParam.get("properties_list") or [])
            if not props:
                raise ValueError("EQdiffParam.context_mapping is true but EQdiffParam.properties_list is empty.")
            cmd += ["--context-mapping",
                    "--num-context-features", str(len(props)),
                    "--properties-list", *map(str, props)]
        if cfg.EQdiffParam.concat_datasets:
            cmd += ["--concat-datasets",
                    "--alpha", str(cfg.EQdiffParam.alpha),
                    "--beta", str(cfg.EQdiffParam.beta)]
            if cfg.EQdiffParam.get("samples_per_epoch") is not None:
                cmd += ["--samples-per-epoch", str(cfg.EQdiffParam.samples_per_epoch)]
            if cfg.EQdiffParam.get("min_original_share") is not None:
                cmd += ["--min-original-share", str(cfg.EQdiffParam.min_original_share)]
        result = subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
        sync_mlflow(cfg)
    # Re-check after training: best_mol_stab.ckpt may only exist now
    save_ckpt = iter_best_ckpt if iter_best_ckpt.exists() else iter_last_ckpt
    if not save_ckpt.exists():
        raise FileNotFoundError(f"Expected model checkpoint not found at {save_ckpt}. Check training logs for errors.")

    # Step 2b (active learning): choose this iteration's targets from the labelled data and the generator's past
    # accuracy. It does not depend on split/train, so it runs right before generation; re-planning (delete
    # targets.json, pipeline.start_at: plan) then does not retrain the model.
    if al_on and 'plan' in active_blocks:
        print(f"  Running Plan targets ({al['query_strategy']}) ")
        from src.pipeline_blocks.plan_targets import run_plan
        run_plan(traj=my_trajectory, al=al, hdf5_path=dm.output_hdf5, iter_dir=dm.current_data, iteration=n,
                 total_samples=int(cfg.EQdiffParam.num_samples), seed=int(cfg.system.seed),
                 state_file=al_paths(cfg)["state_file"], reference_hdf5=dm.input_hdf5)

    # Step 3: Generate molecules
    generated_data_path = dm.current_data / "generated"
    if 'generate' in active_blocks and al_on:
        print("  Running Generate molecules for the planned targets  ")
        plan = read_targets(dm.current_data)
        if plan is None:
            raise FileNotFoundError(f"No targets.json in {dm.current_data}; include the 'plan' block before 'generate'.")
        dm.clean_dir(dm.query_data)
        # with dedup, query/ is filled by the dedup block with the surviving molecules only
        fill_query = 'dedup' not in active_blocks
        cand_meta = {}
        started = time.time()
        for t in plan[1]:
            if cand_meta and (time.time() - started) > float(al["generation_budget_hours"]) * 3600:
                print(f"  Generation budget reached; skipping targets from {t.id} on.")
                break
            target_dir = generated_data_path / f"t{t.id}"
            # a rerun overwrites mol_<k>.xyz by index; clear the folder so no files from an earlier run remain
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target = n_nodes_fallback(cfg, n, my_trajectory.target_args(t.props))
            print(f"  Target {t.id}: {' '.join(target)} x {t.n_samples}")
            cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
                    "--exclusive", "--export=ALL",
                    "python", str(cfg.EQdiff.generate),
                    "--model-path", str(save_ckpt),
                    "--save-dir", str(target_dir),
                    "--save-xyz",
                    "--batch-size", str(cfg.EQdiffParam.batch_size),
                    "--ngraphs", str(t.n_samples),
                    "--generate-ngraphs-valid",
                    *target,
                    ]
            subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
            # query/t<id>_mol_<k>.xyz; candidates_meta.json maps each name back to its target
            for xyz in sorted(target_dir.glob("mol_*.xyz")):
                name = candidate_name(t.id, xyz.stem)
                if fill_query:
                    shutil.copy(xyz, dm.query_data / f"{name}.xyz")
                cand_meta[name] = {"target_id": t.id}
        write_candidates_meta(dm.current_data, cand_meta)
        if not cfg.EQdiffParam.concat_datasets:
            dm.clean_dir(dm.diffusion_data_dir / 'raw')
            dm.clean_dir(dm.diffusion_data_dir / 'processed')
    elif 'generate' in active_blocks:
        print("  Running Generate molecules  ")
        # Find Target for the current iteration
        # --num-nodes comes from the trajectory's atNUM column (if present),
        # --fixed-context from the conditioning properties (query.props).
        target = my_trajectory.get_target(n) if (my_trajectory.has_properties or my_trajectory.has_num_nodes) else []
        # fall back to EQdiffParam.n_nodes; if unset, atom counts are sampled from the training distribution
        target = n_nodes_fallback(cfg, n, target)
        print(f"  Generation target: {' '.join(target) if target else '(unconditioned, sampled atom counts)'}")

        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
                "--exclusive", "--export=ALL",
                "python", str(cfg.EQdiff.generate),
                "--model-path", str(save_ckpt),
                "--save-dir", str(generated_data_path),
                "--save-xyz",
                "--batch-size", str(cfg.EQdiffParam.batch_size),
                "--ngraphs", str(cfg.EQdiffParam.num_samples),
                "--generate-ngraphs-valid",
                *target,
                ]
        result = subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
        # clean temp .ckpt files
        # dm.clean_dir(itr_model_path / "run0", ".ckpt")
        if not cfg.EQdiffParam.concat_datasets:
            dm.clean_dir(dm.diffusion_data_dir / 'raw')
            dm.clean_dir(dm.diffusion_data_dir / 'processed')
        # clean temp generated files .pickle
        shutil.copytree(generated_data_path, dm.query_data, dirs_exist_ok=True)
        # delete all files except .xyz
        dm.clean_dir(dm.query_data, exclude_suffix='.xyz')

    # Step 3b (active learning): remove molecules with duplicate connectivity, and those in the initial dataset.
    # Reads every molecule from generated/t<id>/ (via candidates_meta.json) and writes only the survivors to
    # query/, which the backward step then works on.
    if al_on and 'dedup' in active_blocks:
        print("  Running Dedup (connectivity)  ")
        from src.pipeline_blocks.dedup_molecules import run_dedup
        d = al["dedup"]
        run_dedup(query_dir=dm.query_data, iter_dir=dm.current_data, reference_hdf5=dm.input_hdf5,
                  scope=d.get("scope", "all"), against_initial=bool(d.get("against_initial", True)),
                  cov_factor=float(d.get("cov_factor", 1.3)))


if __name__ == "__main__":
    '''
    Main entry point for the script.
    '''
    main()
 
