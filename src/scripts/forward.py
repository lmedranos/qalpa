import os
from pathlib import Path
import shutil
import subprocess
import torch
from hydra import main as hydra_main
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.data_manage import DataManager
from src.utils.trajectory import Trajectory


# Allow experiment_config.yaml to derive values from list lengths
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

    # Create the intial hdf5 data directory if it doesn't exist (experiment_config.yaml -> data.exp_data_path)
    dm = DataManager(cfg.base_dataset_path, cfg.data.exp_data_path)
    dm.hdf5_appending_flag = bool(cfg.hdf5.appending_flag)
    dm.hdf5_iter_flag = bool(cfg.hdf5.hdf5_iter_flag)
    dm.extra_properties_flag = bool(cfg.hdf5.extra_properties_flag)
    dm.initialize_data_directory()

    # Create exterapolating trajectory for properties and scores (experiment_config.yaml -> query.*)
    prop_names = list(cfg.query.props)
    score_names = list(cfg.query.scores)
    my_trajectory = Trajectory(cfg.query.trajectory_csv, prop_names=prop_names, score_names=score_names)
    # Context properties for training and generation (EQdiffParam.context_mapping / properties_list)
    context_props = my_trajectory.get_context_props(list(cfg.EQdiffParam.properties_list), cfg.EQdiffParam.context_mapping)

    # Current active-learning iteration and loop settings (experiment_config.yaml -> loop.* / EQdiffParam.*)
    n = cfg.loop.current_iter

    if n > cfg.loop.num_iterations:
        raise ValueError(f"Current iteration {n} exceeds the total number of iterations {cfg.loop.num_iterations}.")

    # Pipeline blocks and optional resume point (experiment_config.yaml -> pipeline.blocks / pipeline.start_at)
    # start_at: null runs all blocks; an unknown start_at raises instead of rerunning the whole iteration
    pipeline_blocks = list(cfg.pipeline.blocks)
    start_at = cfg.pipeline.start_at
    if start_at is None:
        active_blocks = set(pipeline_blocks)
    elif start_at in pipeline_blocks:
        active_blocks = set(pipeline_blocks[pipeline_blocks.index(start_at):])
    else:
        raise ValueError(f"pipeline.start_at '{start_at}' is not in pipeline.blocks {pipeline_blocks}")

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

    # Check checkpoint/config consistency before any job is launched. The network is always rebuilt
    # from the checkpoint's own hyperparameters, so its context properties must match the config.
    check_ckpt, check_hint = None, ""
    if 'train' in active_blocks and load_ckpt is not None:
        check_ckpt = Path(load_ckpt)
        check_hint = ("Fine-tuning keeps the checkpoint's architecture. Start from a checkpoint with the same context "
                      "properties, or retrain from iteration 0 with EQdiffParam.load_ckpt_from_pretrained: null.")
    elif 'generate' in active_blocks and save_ckpt.exists():
        check_ckpt = save_ckpt
        check_hint = "Generation would ignore the property targets. Rerun the 'train' block for this iteration."
    if check_ckpt is not None:
        ckpt_hparams = torch.load(check_ckpt, map_location="cpu").get("hyper_parameters", {})
        ckpt_props = list(ckpt_hparams.get("properties_list") or []) if ckpt_hparams.get("context_mapping") else []
        if ckpt_props != context_props:
            raise ValueError(
                f"{check_ckpt} was trained with context properties {ckpt_props or '[] (unconditional)'}, "
                f"but the config asks for {context_props or '[] (unconditional)'}. {check_hint}"
            )

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
        # Model hyperparameters for --conf: keys from src/configs/eqdiff_model.yaml, values from the
        # composed config, so experiment_config.yaml and CLI overrides beat the framework defaults.
        # Flags passed after --conf (batch size, epochs, ...) still win.
        model_conf_path = itr_model_path / "eqdiff_model.yaml"
        model_conf = OmegaConf.load(Path(cfg.code_paths.configs) / "eqdiff_model.yaml")
        for key in model_conf:
            if key in cfg:
                model_conf[key] = cfg[key]
        OmegaConf.save(model_conf, model_conf_path, resolve=True)

        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(cfg.EQdiff.train),
            "--conf", str(model_conf_path),
            "--save-dir", str(itr_model_path),
            "--test-save-dir", str(Path(cfg.EQdiffParam.test_save_dir) / f"model_iter_{n}"),
            "--dataset", str(cfg.EQdiffParam.dataset),
            "--dataset-root", str(Path(cfg.EQdiffParam.dataset_root)),
            "--gpus", str(gpus_per_task),
            "--batch-size", str(cfg.EQdiffParam.batch_size),
            "--num-epochs", str(cfg.EQdiffParam.num_epochs),
            "--test-interval", str(cfg.EQdiffParam.test_interval),
            ]
        if context_props:
            cmd += ["--context-mapping",
                    "--num-context-features", str(len(context_props)),
                    "--properties-list", *context_props]
        if load_ckpt is not None:
            cmd += ["--load-ckpt-from-pretrained", str(load_ckpt)]
        if cfg.EQdiffParam.concat_datasets:
            cmd += ["--concat-datasets",
                    "--alpha", str(cfg.EQdiffParam.alpha),
                    "--beta", str(cfg.EQdiffParam.beta)]
            if cfg.EQdiffParam.get("samples_per_epoch") is not None:
                cmd += ["--samples-per-epoch", str(cfg.EQdiffParam.samples_per_epoch)]
            if cfg.EQdiffParam.get("min_original_share") is not None:
                cmd += ["--min-original-share", str(cfg.EQdiffParam.min_original_share)]
        # MLflow monitoring (optional, experiment_config.yaml -> mlflow.*): import the training logs of all
        # iterations into MLflow with the separate MLflow environment (mlflow.python), every
        # mlflow.sync_interval_minutes while training runs and once more at the end. Sync failures only warn.
        mlcfg = cfg.get("mlflow")
        sync_cmd, sync_interval = None, None
        if mlcfg and mlcfg.get("enabled", False):
            sync_cmd = [
                str(mlcfg.python),
                str(Path(_REPO_ROOT) / "src" / "scripts" / "sync_mlflow.py"),
                "--model-root", str(cfg.EQdiffParam.model_root),
                "--tracking-uri", str(mlcfg.tracking_uri),
                "--experiment", str(cfg.experiment_name),
            ]
            interval_minutes = mlcfg.get("sync_interval_minutes", 10)
            # null or <= 0: sync only once, after training
            sync_interval = 60 * float(interval_minutes) if interval_minutes and interval_minutes > 0 else None

        train_proc = subprocess.Popen(cmd, cwd=_ORIGINAL_CWD, env=os.environ.copy())
        training_done = False
        while not training_done:
            try:
                train_proc.wait(timeout=sync_interval)  # no MLflow: waits until training ends
                training_done = True
            except subprocess.TimeoutExpired:
                pass
            if sync_cmd is not None:
                try:
                    subprocess.run(sync_cmd, check=True, cwd=_ORIGINAL_CWD, timeout=sync_interval)
                except (subprocess.SubprocessError, OSError) as err:
                    print(f"WARNING: MLflow sync failed ({err}); training is unaffected.")
        if train_proc.returncode != 0:
            raise subprocess.CalledProcessError(train_proc.returncode, cmd)
    # Re-check after training: best_mol_stab.ckpt may only exist now
    save_ckpt = iter_best_ckpt if iter_best_ckpt.exists() else iter_last_ckpt
    if not save_ckpt.exists():
        raise FileNotFoundError(f"Expected model checkpoint not found at {save_ckpt}. Check training logs for errors.")

    # Step 3: Generate molecules
    generated_data_path = dm.current_data / "generated"
    if 'generate' in active_blocks:
        print("  Running Generate molecules  ")
        # Find Target for the current iteration
        if my_trajectory.has_properties:
            target = my_trajectory.get_target(n)
        else:
            n_nodes_cfg = cfg.EQdiffParam.get("n_nodes")
            if n_nodes_cfg is None:
                # Molecule sizes are drawn from the training size distribution
                target = []
            elif isinstance(n_nodes_cfg, (list, ListConfig)):
                n_nodes = n_nodes_cfg[n] if len(n_nodes_cfg) > 1 else n_nodes_cfg[0]
                target = ["--num-nodes", str(n_nodes)]
            else:
                target = ["--num-nodes", str(n_nodes_cfg)]

        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
                "--exclusive", "--export=ALL",
                "python", str(cfg.EQdiff.generate),
                "--model-path", str(save_ckpt),
                "--save-dir", str(generated_data_path),
                "--save-xyz",
                "--batch-size", str(cfg.get("inference_batch_size", cfg.EQdiffParam.batch_size)),
                "--ngraphs", str(cfg.EQdiffParam.num_samples),
                "--seed", str(cfg.get("seed", cfg.system.seed)),
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


if __name__ == "__main__":
    '''
    Main entry point for the script.
    '''
    main()

