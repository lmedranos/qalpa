import argparse
import os
import sys
from pathlib import Path

def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    
    # Number of iterations and max parallel tasks
    max_parallel_tasks = args.max_parallel_tasks
    
    # utils directories
    utils_path =Path(args.actdiff_utils)
    print(f"utils_path: {utils_path}")
    sys.path.insert(0,str(utils_path))
    
    from data_manage import DataManager
    from parallelize import node_parallelize
    from trajectory import Trajectory
    
    # Code directories 
    blocks_path =Path(args.pipeline_blocks)
    sys.path.insert(0,str(blocks_path))
    

    # Define FIH-aims executable
    fih_aims_exc = " /software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x"
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(args.base_dataset, args.data)
    dm.initialize_data_directory()
    
    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 

    # Number of models to evaluate
    num_models = args.num_models
    num_targets = len(my_trajectory)   # len(my_trajectory.target_values)
    
    t = num_targets -1 # Set target to the last one for training, since we want to train on the most extrapolated target and evaluate on all targets. This is important for the evaluation of the extrapolation capabilities of the model.
    m = num_models -1 # Set model to the last one for training, since we want to train all models and evaluate them on all targets. This is important for the evaluation of the extrapolation capabilities of the model.
    #for m in range(num_models):
    if True:     
        #for t in range(num_targets):
        if True:
            dm.create_evaluation_directories(t, m)
            print(f"Evaluating target {t} model {m}")
            # # Step 4: Calculate properties 
            prop_calc_path = dm.current_data / "prop_calc"
            # Run FIH-aims
            kfiles = node_parallelize(fih_aims_exc, prop_calc_path, 'aims.out', max_parallel_tasks) 
            print(f"killed files: {kfiles} for target {t} model {m}")
            dm.unlink_mol(kfiles, dm.query_data)

        


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
    parser.add_argument("--num_models", type=int, default=10)
    parser.add_argument("--geom_optimizer", type=str)
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
