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
    # query_data_path = dm.current_data / "query"

    # Step 1: Run data split
    if 'split' in active_blocks:
        print(" Running Data split ")
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(Path(cfg.EQdiff.split)),
            "--file-path", str(hdf5_file_path),
            "--out-path", str(cfg.EQdiffParam.dataset_root),
            ]
        result = subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
        print(result.stdout)
        print(result.stderr)
    # step 2: Diffusion model training
    itr_model_path = Path(cfg.EQdiffParam.model_root) / f"model_iter_{n}"
    itr_model_path.mkdir(parents=True, exist_ok=True)
    
    # the previous iteration's checkpoint on every iteration after that.
    if n > 0:
        pervious_ckpt_dir = Path(cfg.EQdiffParam.model_root) / f"model_iter_{n - 1}" 
        best_ckpt = pervious_ckpt_dir / "best_mol_stab.ckpt"
        last_ckpt = pervious_ckpt_dir / "run0" / "last.ckpt"
        load_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
    elif cfg.EQdiffParam.load_ckpt_from_pretrained not in (None, "None", ""):
        load_ckpt = cfg.EQdiffParam.load_ckpt_from_pretrained
    else:
        load_ckpt = None

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
        if load_ckpt is not None:
            cmd += ["--load-ckpt-from-pretrained", str(load_ckpt)]
        result = subprocess.run(cmd, check=True, cwd=_ORIGINAL_CWD, env=os.environ.copy())
        
    best_ckpt = itr_model_path  / "best_mol_stab.ckpt"
    last_ckpt = itr_model_path / "run0" / "last.ckpt"
    save_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
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
 
