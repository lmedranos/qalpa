import argparse
import os
import sys
import subprocess
from pathlib import Path
import time
import socket
import tarfile
import shutil


def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    
    # utils directories
    utils_path =Path(args.actdiff_utils)
    sys.path.insert(0,str(utils_path))
    from data_manage import DataManager
    from trajectory import Trajectory
    # data directories 
    base_dataset = args.base_dataset
    dataset_name = args.dataset_name
    # Code directories 
    blocks_path =Path(args.pipeline_blocks)
    sys.path.insert(0,str(blocks_path))


    # Diffusion model directories
    model_code_path = Path(args.model_code)
    sys.path.insert(0,str(model_code_path / "e3mol_aqm" / "experiments" / "data"))
    sys.path.insert(0,str(model_code_path / "e3mol_aqm"))
    model_config_path = Path(args.config, "diffusion_aqm.yaml")
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    

    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 

    # Flag to resume in the midde of the loop
    steps={'split', 'train', 'generate', 'relax', 'score_filter', 'fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}
    # Run active learning loop

    num_iterations = args.num_iterations
    n = args.iter

    hdf5_file_path = dm.output_hdf5      # append should be always true for this setup

    print(f"\n Iteration {n}")

    # Setup environment for GNN
    gpus_per_task = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    # env = os.environ.copy()   
    # env["OMP_NUM_THREADS"] = str(cpus_per_task)
    # env["MKL_NUM_THREADS"] = str(cpus_per_task)
    # os.environ.update(env)
    
    # Create directories for the current iteration
    dm.create_iteration_directories(n)


    # Step 1: Run data split
    print(" Running Data split ")
    script_path = model_code_path / "e3mol_aqm" / "experiments" / "data" / "aqm" / "split_data.py"
    cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}",
        "--exclusive", "--export=ALL",
        "python", str(script_path),
        "--file-path", str(hdf5_file_path),
        "--out-path", str(dm.diffusion_data_dir)
        ]
    subprocess.run(cmd, check=True)  if 'split' not in skip_step_atstart else skip_step_atstart.discard('split')

    # step 2: Diffusion model training
    print("  Running Diffusion model training ")
    script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_train.py"
    itr_model_path = dm.diffusion_model_dir / f"model_iter_{n}"
    itr_model_path.mkdir(parents=True, exist_ok=True)
    cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", "--gres=gpu:2",
        "--exclusive", "--export=ALL",
        "python", str(script_path),
        "--conf", str(model_config_path), 
        "--save-dir", str(itr_model_path),
        "--dataset", dataset_name, 
        "--dataset-root", str(dm.diffusion_data_dir),
        "--gpus", str(gpus_per_task),
        "--batch-size", "32"]
    subprocess.run(cmd, check=True, env=os.environ.copy())  if 'train' not in skip_step_atstart else skip_step_atstart.discard('train')
    model_path = itr_model_path / "best_mol_stab.ckpt"
    # Update or just delete the model path in the config file for the next iteration
    update_flag = n < (num_iterations - 1)
    dm.edit_config(model_config_path, "load_ckpt_from_pretrained", str(model_path), update_flag)

    # Step 3: Generate molecules
    print("  Running Generate molecules  ")
    script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_evaluation_qm.py"
    generated_data_path = dm.current_data / "generated" 
    # Find Target for the current iteration
    if my_trajectory.has_properties:
        target = my_trajectory.get_target(n)
    else:
        n_nodes = args.n_nodes[n]  if len(args.n_nodes) > 1 else args.n_nodes[0]
        target = ["--num-nodes", str(n_nodes)]           
    cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(script_path),
            "--model-path", str(model_path),
            "--save-dir", str(generated_data_path),
            "--save-xyz", 
            "--batch-size", "20",
            "--ngraphs", "200",
            "--generate-ngraphs-valid",
            *target,
            ]
    print(f"  Command: {cmd}")
    subprocess.run(cmd, check=True, env=os.environ.copy()) if 'generate' not in skip_step_atstart else skip_step_atstart.discard('generate')

    
    # clean temp .ckpt files
    dm.clean_files(itr_model_path / "run0", ".ckpt")
    dm.clean_files(dm.diffusion_data_dir/'raw')
    dm.clean_files(dm.diffusion_data_dir/'processed')
    
    # clean temp generated files .pickle
    query_data_path = dm.current_data / "query"
    shutil.copytree(generated_data_path, query_data_path, dirs_exist_ok=True)
    #delete all files except .xyz
    for item in query_data_path.iterdir():
        if item.is_file() and item.suffix != '.xyz':
            item.unlink()   
    
    # clean temp generated files .xyz
    # dm.clean_files(generated_data_path, '.xyz')


if __name__ == "__main__":
    '''
    Main entry point for the script.
    Parses command line arguments and starts the main loop.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dataset", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--model_code", type=str)
    parser.add_argument("--pipeline_blocks", type=str)
    parser.add_argument("--actdiff_utils", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--props", type=str, nargs='+', default=[])
    parser.add_argument("--scores", type=str, nargs='+', default=[])
    parser.add_argument("--n_nodes", type=int, nargs='+', default=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    parser.add_argument("--filter_scheme", type=str, nargs='+', default=[])
    parser.add_argument("--max_parallel_tasks", type=int, default=1)
    parser.add_argument("--iter", type=int, default=0)

    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
