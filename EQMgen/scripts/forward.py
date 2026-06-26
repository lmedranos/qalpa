"""
EQMgen.scripts.forward - Forward pass: data generation and molecular generation

This module handles:
1. Data splitting
2. Diffusion model training
3. Molecule generation
"""

import argparse
import os
import sys
from pathlib import Path
import shutil
import subprocess


def forward_main(
    base_dataset: str,
    dataset_name: str,
    data_path: str,
    model_code: str,
    config_path: str,
    actdiff_utils: str,
    num_iterations: int = 10,
    iter: int = 0,
    batch_size: int = 32,
    n_graphs: int = 200,
    props: list = None,
    scores: list = None,
    n_nodes: list = None,
    max_parallel_tasks: int = 1,
):
    """
    Main function to run the forward workflow pass.
    
    Args:
        base_dataset: Path to base HDF5 dataset
        dataset_name: Name of the dataset
        data_path: Path to data directory
        model_code: Path to model code directory
        config_path: Path to config directory
        actdiff_utils: Path to utils directory
        num_iterations: Total number of iterations
        iter: Current iteration number
        batch_size: Batch size for training
        n_graphs: Number of graphs to generate
        props: List of property names
        scores: List of score names
        n_nodes: List of node counts per iteration
        max_parallel_tasks: Maximum parallel tasks
    """
    if props is None:
        props = []
    if scores is None:
        scores = []
    if n_nodes is None:
        n_nodes = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    
    # Setup environment for GNN
    gpus_per_task = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
 
    # utils directories
    utils_path = Path(actdiff_utils)
    sys.path.insert(0, str(utils_path))
    from data_manage import DataManager
    from trajectory import Trajectory
    
    # data directories 
    base_dataset_path = Path(base_dataset)
    
    # Diffusion model directories
    model_code_path = Path(model_code)
    sys.path.insert(0, str(model_code_path / "e3mol_aqm" / "experiments" / "data"))
    sys.path.insert(0, str(model_code_path / "e3mol_aqm"))
    model_config_path = Path(config_path) / "diffusion_aqm.yaml"

    # Create the initial hdf5 data directory if it doesn't exist
    dm = DataManager(str(base_dataset_path), data_path)
    dm.initialize_data_directory()
    if dm.hdf5_appending_flag:
        hdf5_file_path = dm.output_hdf5
    else:
        hdf5_file_path = dm.input_hdf5 
    
    # Set diffusion data and log directories in the yaml configuration file 
    dm.edit_config(model_config_path, "dataset_root", str(dm.diffusion_data_dir))
    dm.edit_config(model_config_path, "save_dir", str(dm.diffusion_log_dir))
    dm.edit_config(model_config_path, "test_save_dir", str(dm.diffusion_test_dir))
    dm.edit_config(model_config_path, "load_ckpt_from_pretrained", '', False)
 
    # Create extrapolating trajectory for properties and scores
    my_trajectory = Trajectory(str(config_path), prop_names=props, score_names=scores) 
    context_flag = True if my_trajectory.has_properties else False
    dm.edit_config(model_config_path, "context_mapping", str(context_flag))
    dm.edit_config(model_config_path, "properties_list", [p for p in my_trajectory.prop_names if p != "atNUM"])   # Exclude 'atNUM' from properties list in the config file
    dm.edit_config(model_config_path, "num_context_features", str(len(my_trajectory.prop_names) - (1 if "atNUM" in my_trajectory.prop_names else 0)))  # Exclude 'atNUM' from context features
    
    # Flag to resume in the middle of the loop
    skip_step_atstart = set()
    
    # Create directories for the current iteration
    dm.create_iteration_directories(iter)
    
    # Step 1: Run data split
    script_path = model_code_path / "e3mol_aqm" / "experiments" / "data" / "aqm" / "split_data.py"
    if 'split' not in skip_step_atstart:
        print(" Running Data split ")
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(script_path),
            "--file-path", str(hdf5_file_path),
            "--out-path", str(dm.diffusion_data_dir)
            ]
        subprocess.run(cmd, check=True)  
    
    # Step 2: Diffusion model training
    script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_train.py"
    itr_model_path = dm.diffusion_model_dir / f"model_iter_{iter}"
    itr_model_path.mkdir(parents=True, exist_ok=True)
    model_path = itr_model_path / "best_mol_stab.ckpt"
    if 'train' not in skip_step_atstart:
        print("  Running Diffusion model training ")
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(script_path),
            "--conf", str(model_config_path), 
            "--save-dir", str(itr_model_path),
            "--dataset", dataset_name, 
            "--dataset-root", str(dm.diffusion_data_dir),
            "--gpus", str(gpus_per_task),
            "--batch-size", str(batch_size)]
        subprocess.run(cmd, check=True, env=os.environ.copy()) 
        # Update or just delete the model path in the config file for the next iteration
        update_flag = iter < (num_iterations - 1)
        dm.edit_config(model_config_path, "load_ckpt_from_pretrained", str(model_path), update_flag)
    
    # Step 3: Generate molecules
    script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_evaluation_qm.py"
    generated_data_path = dm.current_data / "generated" 
    if 'generate' not in skip_step_atstart:
        print("  Running Generate molecules  ")
        # Find Target for the current iteration
        if my_trajectory.has_properties:
            target = my_trajectory.get_target(iter)
        else:
            n_nodes_val = n_nodes[iter] if len(n_nodes) > 1 else n_nodes[0]
            target = ["--num-nodes", str(n_nodes_val)]           
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
                "--exclusive", "--export=ALL",
                "python", str(script_path),
                "--model-path", str(model_path),
                "--save-dir", str(generated_data_path),
                "--save-xyz", 
                "--batch-size", str(batch_size),
                "--ngraphs", str(n_graphs),
                "--generate-ngraphs-valid",
                *target,
                ]
        subprocess.run(cmd, check=True, env=os.environ.copy()) 
        # clean temp .ckpt files
        dm.clean_dir(itr_model_path / "run0", ".ckpt")
        dm.clean_dir(dm.diffusion_data_dir / 'raw')
        dm.clean_dir(dm.diffusion_data_dir / 'processed')
        # Copy generated files to query data
        shutil.copytree(generated_data_path, dm.query_data, dirs_exist_ok=True)
        # Keep only .xyz files
        dm.clean_dir(dm.query_data, exclude_suffix='.xyz')
    
    print(f"\n Completed forward pass for iteration {iter} \n")


if __name__ == "__main__":
    '''
    Main entry point for the script when run as CLI.
    Parses command line arguments and starts the forward pass.
    '''
    parser = argparse.ArgumentParser(description="Forward pass: Generate molecules using diffusion model")
    parser.add_argument("--base_dataset", type=str, required=True, help="Path to base HDF5 dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--model_code", type=str, required=True, help="Path to model code")
    parser.add_argument("--config", type=str, required=True, help="Path to config directory")
    parser.add_argument("--actdiff_utils", type=str, required=True, help="Path to utils")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_graphs", type=int, default=200, help="Number of graphs to generate")
    parser.add_argument("--num_iterations", type=int, default=10, help="Total iterations")
    parser.add_argument("--iter", type=int, default=0, help="Current iteration")
    parser.add_argument("--props", type=str, nargs='+', default=[], help="Property names")
    parser.add_argument("--scores", type=str, nargs='+', default=[], help="Score names")
    parser.add_argument("--n_nodes", type=int, nargs='+', default=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110], help="Nodes per iteration")
    parser.add_argument("--max_parallel_tasks", type=int, default=1, help="Max parallel tasks")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    forward_main(
        base_dataset=args.base_dataset,
        dataset_name=args.dataset_name,
        data_path=args.data,
        model_code=args.model_code,
        config_path=args.config,
        actdiff_utils=args.actdiff_utils,
        num_iterations=args.num_iterations,
        iter=args.iter,
        batch_size=args.batch_size,
        n_graphs=args.n_graphs,
        props=args.props,
        scores=args.scores,
        n_nodes=args.n_nodes,
        max_parallel_tasks=args.max_parallel_tasks,
    )
