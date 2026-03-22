import argparse
import os
import sys
import subprocess
from pathlib import Path
import shutil

#from pipeline_blocks.to_hdf5 import run_to_hdf5


def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    
    # --------------------------------------------------------------------------------
    #                               -- Intializations --
    # --------------------------------------------------------------------------------
    
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
    from to_hdf5 import run_to_hdf5

    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    context_flag = True if my_trajectory.has_properties else False

    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    context_flag = True  

    # Run active learning loop
    num_models = args.num_models
    num_targets = len(my_trajectory) 

    for m in range(num_models):     
        for t in range(num_targets):
            print(f"creating extra properties files for model {m} and target {t}")
            dm.create_evaluation_directories(t, m)
            prop_calc_path = dm.current_data / "prop_calc"
            (dm.current_data / "hdf5").mkdir(parents=True, exist_ok=True)
            output_hdf5 = dm.current_data / "hdf5" / f"m{m}-t{t}-output.hdf5"
            extra_prop_hdf5 = dm.current_data / "hdf5" / f"m{m}-t{t}-extra_properties.hdf5"
            run_to_hdf5(dm.query_data, prop_calc_path, dm.hdf5_appending_flag, output_hdf5, context_flag, extra_prop_hdf5, dm.itr_hdf5_path, dm.itr_extra_prop_path, "eval") 
            
 
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
    parser.add_argument("--geom_optimizer", type=str)
    parser.add_argument("--num_models", type=int, default=10)

    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
