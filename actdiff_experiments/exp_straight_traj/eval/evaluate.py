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
    from parallelize import node_parallelize
    from trajectory import Trajectory
    
    # Code directories 
    blocks_path =Path(args.pipeline_blocks)
    sys.path.insert(0,str(blocks_path))
    from filter_molecules import MolecularFilter
    from propcalc_postprocess import postprocess_fhi
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(args.base_dataset, args.data)
    dm.initialize_data_directory()
    
    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    context_flag = True if my_trajectory.has_properties else False
    
    # Number of models to evaluate
    num_models = args.num_models
    num_targets = len(my_trajectory)   # len(my_trajectory.target_values)
    
    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = args.filter_scheme

    # Flag to resume in the midde of the loop
    steps={'score_filter', 'fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}
        
    t = num_targets -1 # Set target to the last one for training, since we want to train on the most extrapolated target and evaluate on all targets. This is important for the evaluation of the extrapolation capabilities of the model.
    m = num_models -1 # Set model to the last one for training, since we want to train all models and evaluate them on all targets. This is important for the evaluation of the extrapolation capabilities of the model.
    #for m in range(num_models):
    if True:     
        #for t in range(num_targets):
        if True:   
            dm.create_evaluation_directories(t, m)
            query_data_path = dm.current_data / "query"

            # # Step 4: Calculate properties 
            prop_calc_path = dm.current_data / "prop_calc"
            if context_flag and 'post_fih_aims' not in skip_step_atstart:
                print("  Running Calculate properties postprocessing ")
                # Postprocess FIH-aims output to create json files
                for f in dm.query_data.iterdir():
                    postprocess_fhi(f, prop_calc_path)
            elif context_flag:
                skip_step_atstart.discard('post_fih_aims')
            else:
                print("Calculating properties postprocessing is skipped.")

            
            # Step 5: Calculate accuracy based on properties
            if 'acc_filter' not in skip_step_atstart:
                if 'accuracy' in filteration_scheme and my_trajectory.has_properties:
                    print("  Running accuracy molecules Filter based on properties  ")
                    result_path = dm.current_data / "results"
                    for trajprop in my_trajectory.prop_names:
                        if trajprop.lower() == 'atnum':
                            continue
                        print(f"    Accuracy based on {trajprop}  ")
                        for f in dm.query_data.iterdir():
                            if f.is_file() and f.suffix == '.xyz': 
                                molecular_filter.accuracy_calculation(f, dm.results, trajprop, prop_calc_path, t)           
            else:
                skip_step_atstart.discard('acc_filter')
                
            print(f"Evaluation target: {t}, model: {m} is done.")


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
 
