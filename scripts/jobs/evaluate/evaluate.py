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
    
    # data directories 
    base_dataset = args.base_dataset
    
    # Code directories 
    blocks_path =Path(args.pipeline_blocks)
    sys.path.insert(0,str(blocks_path))
    from filter_molecules import MolecularFilter
    from calculate_sascore import sascore
    from propcalc_postprocess import postprocess_fhi
    from propcalc_preprocess import preprocess_fhi
    
   
    # Define FIH-aims executable
    fih_aims_exc = " /software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x"
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    

    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    context_flag = True if my_trajectory.has_properties else False
    num_models = args.num_models

    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = args.filter_scheme

    # Flag to resume in the midde of the loop
    steps={'score_filter', 'fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}
        
    for t in range(len(my_trajectory)): 
        for m in range(num_models):    
            dm.create_evaluation_directories(t, m)
            query_data_path = dm.current_data / "query"

            # Step 3: Calculate Score 
            if 'score_filter' not in skip_step_atstart:
                if 'score' in filteration_scheme and my_trajectory.has_scores:
                    print("  Running Calculating Score  ")
                    result_path = dm.current_data / "results"
                    for score_name in my_trajectory.score_names:
                        print(f"    Calculation: {score_name}  ")
                        for f in query_data_path.iterdir():
                            if f.is_file() and f.suffix == '.xyz':
                                if score_name.lower() == 'sascore':
                                    sascore(f, result_path)
                                # elif score_name.lower() == 'qed':  # To be implemented
            else:
                skip_step_atstart.discard('score_filter')


            # Step 4: Calculate properties 
            prop_calc_path = dm.current_data / "prop_calc"
            if context_flag and 'fih_aims' not in skip_step_atstart:
                print("  Running Calculate properties ")
                # Prepare input files for FIH-aims
                for f in query_data_path.iterdir():
                    preprocess_fhi(f, prop_calc_path, Path(os.environ["CONFIG"],"nctr1.in"))
                # Run FIH-aims
                node_parallelize(fih_aims_exc, prop_calc_path, 'aims.out', max_parallel_tasks) 
                # Postprocess FIH-aims output to create json files
                for f in prop_calc_path.iterdir():
                    postprocess_fhi(f, prop_calc_path)
            elif context_flag:
                skip_step_atstart.discard('fih_aims')
            else:
                print("Calculating properties is skipped.")

            
            # Step 5: Calculate accuracy based on properties
            if 'acc_filter' not in skip_step_atstart:
                if 'accuracy' in filteration_scheme and my_trajectory.has_properties:
                    print("  Running accuracy molecules Filter based on properties  ")
                    result_path = dm.current_data / "results"
                    for trajprop in my_trajectory.prop_names:
                        if trajprop.lower() == 'atnum':
                            continue
                        print(f"    Accuracy based on {trajprop}  ")
                        for f in query_data_path.iterdir():
                            if f.is_file() and f.suffix == '.xyz': 
                                molecular_filter.accuracy_calculation(f, result_path, trajprop, prop_calc_path, t)
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

    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
