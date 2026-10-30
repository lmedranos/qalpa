import os
import time
from pathlib import Path
import shutil
import subprocess
from hydra import main as hydra_main
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.data_manage import DataManager
from src.utils.trajectory import Trajectory
from src.active_learning.config import al_paths, get_al_cfg, resolve_blocks, resolve_start, should_stop, validate_al_cfg
from src.active_learning.ids import candidate_name, read_targets, write_candidates_meta
from src.active_learning.plan_block import run_plan


# Allow experiment_config.yaml to derive values such as num_context_features from list lengths
OmegaConf.register_new_resolver("len", len, replace=True)

# Resolve the repository root based on this file's location so that paths in
# global_config.yaml work regardless of the current working directory.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
OmegaConf.register_new_resolver("repo_root", lambda: _REPO_ROOT, replace=True)

# Capture the original working directory before Hydra changes it
_ORIGINAL_CWD = os.getcwd()


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

    if should_stop(cfg):
        print("loop.stop is set; nothing to do.")
        return
    al = get_al_cfg(cfg)
    al_on = al["enabled"]
    al_files = al_paths(cfg)

    # Create the intial hdf5 data directory if it doesn't exist (experiment_config.yaml -> data.exp_data_path)
    dm = DataManager(cfg.base_dataset_path, cfg.data.exp_data_path)
    dm.initialize_data_directory()

    # HDF5 source (experiment_config.yaml -> hdf5.appending_flag, resolved inside DataManager)
    if cfg.hdf5.appending_flag:
        hdf5_file_path = dm.output_hdf5
    else:
        hdf5_file_path = dm.input_hdf5

    # Create exterapolating trajectory for properties and scores (experiment_config.yaml -> query.*)
    prop_names = list(cfg.query.props)
    score_names = list(cfg.query.scores)
    my_trajectory = Trajectory(cfg.query.trajectory_csv, prop_names=prop_names, score_names=score_names)

    if al_on:
        validate_al_cfg(al, int(cfg.EQdiffParam.num_samples), prop_names)
        if not my_trajectory.has_properties:
            raise ValueError("Active learning needs property targets in the trajectory (query.props).")
        if al["query_strategy"] == "kde_map" and len(my_trajectory) < 2:
            raise ValueError("al.query_strategy=kde_map needs at least 2 trajectory rows to define a corridor.")

    # Current active-learning iteration and loop settings (experiment_config.yaml -> loop.* / EQdiffParam.*)
    n = cfg.loop.current_iter
    
    if n > cfg.loop.num_iterations:
        raise ValueError(f"Current iteration {n} exceeds the total number of iterations {cfg.loop.num_iterations}.")

    # Pipeline blocks and optional resume point (eqmgen_paths.yaml -> pipeline.blocks / pipeline.start_at)
    pipeline_blocks = resolve_blocks(cfg)
    start_at = resolve_start(cfg, pipeline_blocks)
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

    # Step 0 (active learning): choose this iteration's targets from the labelled data and the generator's past accuracy
    if al_on and 'plan' in active_blocks:
        print("  Running Plan targets  ")
        run_plan(traj=my_trajectory, al=al, hdf5_path=hdf5_file_path, iter_dir=dm.current_data, iteration=n,
                 total_samples=int(cfg.EQdiffParam.num_samples) * int(al["oversample"]), seed=int(cfg.system.seed),
                 state_file=al_files["state_file"], validation_file=al_files["validation_file"])


    # Step 1: Run data split
    if 'split' in active_blocks:
        print(" Running Data split ")
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(Path(cfg.EQdiff.split)),
            "--file-path", str(hdf5_file_path),
            "--out-path", str(cfg.EQdiffParam.dataset_root),
            ]
        if al_on:
            cmd += ["--seed", str(cfg.system.seed)]
            if al_files["validation_file"].exists():
                cmd += ["--validation-ids", str(al_files["validation_file"])]
        result = subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
        print(result.stdout)
        print(result.stderr)
        
    # step 2: Diffusion model training
    if 'train' in active_blocks:
        print("  Running Diffusion model training ")
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(cfg.EQdiff.train),
            "--conf", str(Path(cfg.code_paths.configs) / "eqdiff_model.yaml"),
            "--save-dir", str(itr_model_path),
            "--dataset", str(cfg.EQdiffParam.dataset),
            "--dataset-root", str(Path(cfg.EQdiffParam.dataset_root)),
            "--gpus", str(gpus_per_task),
            "--batch-size", str(cfg.EQdiffParam.batch_size),
            "--num-epochs", str(cfg.EQdiffParam.num_epochs),
            ]
        # The generator only accepts --fixed-context when trained with these conditioning arguments
        cond_props = [p for p in cfg.EQdiffParam.properties_list if p != "atNUM"]
        if cfg.EQdiffParam.context_mapping and cond_props:
            cmd += ["--context-mapping", "--properties-list", *cond_props, "--num-context-features", str(len(cond_props))]
        if load_ckpt is not None:
            cmd += ["--load-ckpt-from-pretrained", str(load_ckpt)]
        result = subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
    # Training may have just created best_mol_stab.ckpt, so look it up again
    save_ckpt = iter_best_ckpt if iter_best_ckpt.exists() else iter_last_ckpt
    if not save_ckpt.exists():
        raise FileNotFoundError(f"Expected model checkpoint not found at {save_ckpt}. Check training logs for errors.")

    # Step 3: Generate molecules
    generated_data_path = dm.current_data / "generated"
    if 'generate' in active_blocks and al_on:
        print("  Running Generate molecules for the planned targets  ")
        plan = read_targets(dm.current_data)
        if plan is None:
            raise FileNotFoundError(f"No targets.json in {dm.current_data}; include the 'plan' block before 'generate'.")
        dm.clean_dir(dm.query_data)
        cand_meta = {}
        started = time.time()
        for target in plan[1]:
            if cand_meta and (time.time() - started) > float(al["generation_budget_hours"]) * 3600:
                print(f"  Generation budget reached; skipping targets from {target.id} on.")
                break
            target_dir = generated_data_path / f"t{target.id}"
            cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
                    "--exclusive", "--export=ALL",
                    "python", str(cfg.EQdiff.generate),
                    "--model-path", str(save_ckpt),
                    "--save-dir", str(target_dir),
                    "--save-xyz",
                    "--batch-size", str(cfg.EQdiffParam.batch_size),
                    "--ngraphs", str(target.n_samples),
                    "--generate-ngraphs-valid",
                    *my_trajectory.target_args(target.props),
                    ]
            subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
            for xyz in sorted(target_dir.glob("mol_*.xyz")):
                name = candidate_name(target.id, xyz.stem)
                shutil.copy(xyz, dm.query_data / f"{name}.xyz")
                cand_meta[name] = {"target_id": target.id}
        write_candidates_meta(dm.current_data, cand_meta)
        dm.clean_dir(dm.diffusion_data_dir / 'raw')
        dm.clean_dir(dm.diffusion_data_dir / 'processed')
    elif 'generate' in active_blocks:
        print("  Running Generate molecules  ")
        # Find Target for the current iteration
        if my_trajectory.has_properties:
            target = my_trajectory.get_target(n)
        else:
            n_nodes_cfg = cfg.EQdiffParam.n_nodes
            if isinstance(n_nodes_cfg, (list, ListConfig)):
                n_nodes = n_nodes_cfg[n] if len(n_nodes_cfg) > 1 else n_nodes_cfg[0]
            else:
                n_nodes = n_nodes_cfg
            target = ["--num-nodes", str(n_nodes)]

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
        dm.clean_dir(dm.diffusion_data_dir / 'raw')
        dm.clean_dir(dm.diffusion_data_dir / 'processed')
        # clean temp generated files .pickle
        shutil.copytree(generated_data_path, dm.query_data, dirs_exist_ok=True)
        # delete all files except .xyz
        dm.clean_dir(dm.query_data, exclude_suffix='.xyz')


if __name__ == "__main__":
    '''
    Main entry point for the script.
    '''
    main()
 
