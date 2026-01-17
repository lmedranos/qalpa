import argparse
import os
import sys
from pathlib import Path
import shutil
import subprocess


def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    # Setup environment for GNN
    gpus_per_task = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
 
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
    from optimize_geometry import relax_molecule
    from calculate_sascore import sascore
    from filter_molecules import MolecularFilter
    from propcalc_preprocess import preprocess_fhi
    
    # Diffusion model directories
    model_code_path = Path(args.model_code)
    sys.path.insert(0,str(model_code_path / "e3mol_aqm" / "experiments" / "data"))
    sys.path.insert(0,str(model_code_path / "e3mol_aqm"))
    model_config_path = Path(args.config, "diffusion_aqm.yaml")

    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    if dm.hdf5_appending_flag:
        hdf5_file_path = dm.output_hdf5
    else:
        hdf5_file_path = dm.input_hdf5 
    
    # Set diffusion data and log directories in the ymal configuration file 
    dm.edit_config(model_config_path, "dataset_root", str(dm.diffusion_data_dir))
    dm.edit_config(model_config_path, "save_dir", str(dm.diffusion_log_dir))
    dm.edit_config(model_config_path, "test_save_dir", str(dm.diffusion_test_dir))
    dm.edit_config(model_config_path, "load_ckpt_from_pretrained", '', False)
 
    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    context_flag = True if my_trajectory.has_properties else False
    dm.edit_config(model_config_path, "context_mapping", str(context_flag))
    dm.edit_config(model_config_path, "properties_list", [p for p in my_trajectory.prop_names if p != "atNUM"])   # Exclude 'atNUM' from properties list in the config file
    dm.edit_config(model_config_path, "num_context_features", str(len(my_trajectory.prop_names)- (1 if "atNUM" in my_trajectory.prop_names else 0)))  # Exclude 'atNUM' from context features in the config file
    
    # Select geometry optimizer
    if args.geom_optimizer=="SO3LR":
        opt_model = args.geom_optimizer
    elif args.geom_optimizer=="MACE":
        opt_model = Path(os.environ["CONFIG"], "MACE_model_swa.model")
    elif args.geom_optimizer is None:
        opt_model = None
    
    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = args.filter_scheme
       
    # Flag to resume in the midde of the loop
    steps={'split', 'train', 'generate', 'relax', 'score_filter', 'pre_fih_aims'}
    skip_step_atstart = {}
    
    # Run active learning loop
    num_iterations = args.num_iterations
    
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
    else:
        skip_step_atstart.discard('split')
        
    for n in range(num_iterations):
        print(f"\n Iteration {n}")
  
        # step 2: Diffusion model training
        script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_train.py"
        itr_model_path = dm.diffusion_model_dir / f"model_iter_{n}"
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
                "--batch-size", "32"]
            subprocess.run(cmd, check=True, env=os.environ.copy()) 
            # Update or just delete the model path in the config file for the next iteration
            update_flag = n < (num_iterations - 1)
            dm.edit_config(model_config_path, "load_ckpt_from_pretrained", str(model_path), update_flag)
        else:
            skip_step_atstart.discard('train')
            

 
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
    parser.add_argument("--geom_optimizer", type=str)
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
