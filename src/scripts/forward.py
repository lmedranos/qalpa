import os
from pathlib import Path
import shutil
import subprocess
from hydra import main as hydra_main
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.utils.data_manage import DataManager
from src.utils.trajectory import Trajectory


# Allow experiment_config.yaml to derive values such as num_context_features from list lengths
OmegaConf.register_new_resolver("len", len, replace=True)

# Resolve the repository root based on this file's location so that paths in
# global_config.yaml work regardless of the current working directory.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
OmegaConf.register_new_resolver("repo_root", lambda: _REPO_ROOT, replace=True)

# Capture the original working directory before Hydra changes it
_ORIGINAL_CWD = os.getcwd()


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
    score_names = list(cfg.query.scores)
    my_trajectory = Trajectory(cfg.query.trajectory_csv, prop_names=prop_names, score_names=score_names)

    # Current active-learning iteration and loop settings (experiment_config.yaml -> loop.* / EQdiffParam.*)
    n = cfg.loop.current_iter
    
    if n > cfg.loop.num_iterations:
        raise ValueError(f"Current iteration {n} exceeds the total number of iterations {cfg.loop.num_iterations}.")

    # Pipeline blocks and optional resume point (eqmgen_paths.yaml -> pipeline.blocks / pipeline.start_at)
    pipeline_blocks = list(cfg.pipeline.blocks)
    start_at = cfg.pipeline.start_at
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
            "--batch-size", str(cfg.EQdiffParam.batch_size),
            "--num-epochs", str(cfg.EQdiffParam.num_epochs),
            "--test-interval", str(cfg.EQdiffParam.test_interval),
            ]
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
        result = subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
        sync_mlflow(cfg)
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
 
