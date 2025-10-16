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

    # Diffusion model directories
    model_code_path = Path(args.model_code)
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    
    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    num_models = args.num_models

    # Select geometry optimizer
    if args.geom_optimizer=="SO3LR":
        opt_model = args.geom_optimizer
    elif args.geom_optimizer=="MACE":
        opt_model = Path(os.environ["CONFIG"], "MACE_model_swa.model")
            

    # Setup environment for GNN
    gpus_per_task = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    
    # Step 1: Run data split
    if dm.hdf5_appending_flag:
        hdf5_file_path = dm.output_hdf5
    else:
        hdf5_file_path = dm.input_hdf5  
    script_path = model_code_path / "e3mol_aqm" / "experiments" / "data" / "aqm" / "split_data.py"
    print(" Running Data split ")
    cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}",
        "--exclusive", "--export=ALL",
        "python", str(script_path),
        "--file-path", str(hdf5_file_path),
        "--out-path", str(dm.diffusion_data_dir)
        ]
    subprocess.run(cmd, check=True) 
    
       
    for t in range(len(my_trajectory)):
        # Find Target for the current iteration
        if my_trajectory.has_properties:
            target = my_trajectory.get_target(t)
        else:
            n_nodes = args.n_nodes[t]  if len(args.n_nodes) > 1 else args.n_nodes[0]
            target = ["--num-nodes", str(n_nodes)]           
     
        for m in range(num_models):
            dm.create_evaluation_directories(t, m)
            # Step 2: Generate molecules 
            model_path = dm.diffusion_model_dir / f"model_iter_{m}" / "best_mol_stab.ckpt"
            print("  Running Generate molecules  ")
            script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_evaluation_qm.py"
            generated_data_path = dm.current_data / "generated"
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
        
            # clean temp generated files .pickle
            query_data_path = dm.current_data  / "query"
            shutil.copytree(generated_data_path, query_data_path, dirs_exist_ok=True)
            #delete all files except .xyz
            for item in query_data_path.iterdir():
                if item.is_file() and item.suffix != '.xyz':
                    item.unlink()   

            # Step 4: Relax molecules
            print("  Running Relax molecules  ")
            script_path = blocks_path / "relax_molecules.py"
            relaxed_data_path = dm.current_data  / "relaxed"
            for f in query_data_path.iterdir():
                if f.is_file() and f.suffix == '.xyz':
                    relax_molecule(f, relaxed_data_path, opt_model)              
            # clean query directory
            dm.clean_files(query_data_path)
            # Copy all relaxed molecules to the query data path
            shutil.copytree(relaxed_data_path, query_data_path, dirs_exist_ok=True)


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
    parser.add_argument("--geom_optimizer", type=str, default="SO3LR")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
