import argparse
import os
import sys
import subprocess
from pathlib import Path
import shutil


def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
        
    # --------------------------------------------------------------------------------
    #                               -- Intializations --
    # --------------------------------------------------------------------------------
     
    # Setup environment for GNN
    gpus_per_task = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
   
    # utils directories
    utils_path =Path(args.actdiff_utils)
    sys.path.insert(0,str(utils_path))
    from data_manage import DataManager
    from trajectory import Trajectory

    # Diffusion model directories
    model_code_path = Path(args.model_code)
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(args.base_dataset, args.data)
    dm.initialize_data_directory()
    if dm.hdf5_appending_flag:
        hdf5_file_path = dm.output_hdf5
    else:
        hdf5_file_path = dm.input_hdf5 
    
    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
        
    # Flag to resume in the midde of the loop
    steps={'split', 'train', 'generate', 'relax', 'score_filter', 'pre_fih_aims', 'fih_aims', 'post_fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}
    
    # Number of models to evaluate
    num_models = args.num_models
    num_targets = len(my_trajectory)      
          
          
    # Step 0: Run data split
    if 'split' not in skip_step_atstart:
        script_path = model_code_path / "e3mol_aqm" / "experiments" / "data" / "aqm" / "split_data.py"
        print(" Running Data split ")
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(script_path),
            "--file-path", str(hdf5_file_path),
            "--out-path", str(dm.diffusion_data_dir)
            ]
        subprocess.run(cmd, check=True) 
    else:
        skip_step_atstart.discard('split')
           
    for m in range(num_models):     
        for t in range(num_targets):
            # Find Target for the current iteration
            if my_trajectory.has_properties:
                target = my_trajectory.get_target(t)
            else:
                n_nodes = args.n_nodes[t]  if len(args.n_nodes) > 1 else args.n_nodes[0]
                target = ["--num-nodes", str(n_nodes)]           
 
            # --------------------------------------------------------------------------------
            #                       -- Starting new iteration --
            # --------------------------------------------------------------------------------
                    
            # Create directories for the current iteration
            dm.create_evaluation_directories(t, m)
            
            # Step 1: Generate molecules
            generated_data_path = dm.current_data / "generated"
            if 'generate' not in skip_step_atstart: 
                model_path = dm.diffusion_model_dir / f"model_iter_{m}" / "best_mol_stab.ckpt"
                print("  Running Generate molecules  ")
                script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_evaluation_qm.py"
                cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
                        "--exclusive", "--export=ALL",
                        "python", str(script_path),
                        "--model-path", str(model_path),
                        "--save-dir", str(generated_data_path),
                        "--save-xyz", 
                        "--batch-size", "25",
                        "--ngraphs", "50",
                        "--generate-ngraphs-valid",
                        *target,
                        ]
                subprocess.run(cmd, check=True, env=os.environ.copy()) 
                # copy generated molecules to the query data path
                shutil.copytree(generated_data_path, dm.query_data, dirs_exist_ok=True)
                #delete all files except .xyz
                dm.clean_dir(dm.query_data, exclude_suffix='.xyz')
                # clean temp generated files .xyz
                # dm.clean_dir(generated_data_path, '.xyz')
            else:
                skip_step_atstart.discard('generate')


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
    parser.add_argument("--props", type=str, nargs='+', default=[])
    parser.add_argument("--scores", type=str, nargs='+', default=[])
    parser.add_argument("--n_nodes", type=int, nargs='+', default=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    parser.add_argument("--filter_scheme", type=str, nargs='+', default=[])
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--geom_optimizer", type=str)
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
