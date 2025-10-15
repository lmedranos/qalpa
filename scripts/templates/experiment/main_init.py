import argparse
import os
import sys
from pathlib import Path



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

    # Diffusion model directories
    model_config_path = Path(args.config, "diffusion_aqm.yaml")
    

    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    
    # Set diffusion data and log directories in the ymal configuration file 
    dm.edit_config(model_config_path, "dataset_root", str(dm.diffusion_data_dir))
    dm.edit_config(model_config_path, "save_dir", str(dm.diffusion_log_dir))
    dm.edit_config(model_config_path, "test_save_dir", str(dm.diffusion_test_dir))


    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    context_flag = True if my_trajectory.has_properties else False
    dm.edit_config(model_config_path, "context_mapping", str(context_flag))
    dm.edit_config(model_config_path, "properties_list", [p for p in my_trajectory.prop_names if p != "atNUM"])   # Exclude 'atNUM' from properties list in the config file
    dm.edit_config(model_config_path, "num_context_features", str(len(my_trajectory.prop_names)- (1 if "atNUM" in my_trajectory.prop_names else 0)))  # Exclude 'atNUM' from context features in the config file
    

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
    parser.add_argument("--geom_optimizer", type=int, default='MACE')
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
