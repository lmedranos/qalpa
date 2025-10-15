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
    from to_hdf5 import run_to_hdf5
   
    # Define FIH-aims executable
    fih_aims_exc = " /software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x"
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()

    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    context_flag = True if my_trajectory.has_properties else False

    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = args.filter_scheme

    # compress flag to compress iteration files
    compress_flag = False

    # Flag to resume in the midde of the loop
    steps={'relax', 'score_filter', 'fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}
    
    
    n = args.iter    
    # Create directories for the current iteration
    dm.create_iteration_directories(n)
    query_data_path = dm.current_data / "query"

    # Step 5: Calculate Score and Filter molecules 
    if 'score_filter' not in skip_step_atstart:
        if 'score' in filteration_scheme and my_trajectory.has_scores:
            print("  Running Filter molecules based on Score  ")
            score_filtered_path = dm.current_data / "score_filtered"
            for score_name in my_trajectory.score_names:
                print(f"    Filtering based on {score_name}  ")
                for f in query_data_path.iterdir():
                    if f.is_file() and f.suffix == '.xyz':
                        if score_name.lower() == 'sascore':
                            sascore(f, score_filtered_path)
                        # elif score_name.lower() == 'qed':  # To be implemented
                # Filter molecules based on SAScore
                fscored_path = score_filtered_path / score_name
                fscored_path.mkdir(parents=True, exist_ok=True)
                molecular_filter.score_filter(
                    input_path=query_data_path,
                    output_path=fscored_path,
                    score_name=score_name,
                    score_file= score_filtered_path / f'{score_name.lower()}.txt',
                    iter=n
                )
                # clean query directory
                dm.clean_files(query_data_path)
                # copy filtered molecules to the query data path
                shutil.copytree(fscored_path, query_data_path, dirs_exist_ok=True)
                # # clean temp score filtered files .xyz
                # dm.clean_files(fscored_path, '.xyz')
    else:
        skip_step_atstart.discard('score_filter')


    # Step 6: Calculate properties 
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

    
    # Step 7: Filter molecules based on properties
    if 'acc_filter' not in skip_step_atstart:
        if 'accuracy' in filteration_scheme and my_trajectory.has_properties:
            print("  Running Filter molecules based on properties  ")
            accuracy_filtered_path = dm.current_data / "accuracy_filtered"
            for trajprop in my_trajectory.prop_names:
                if trajprop.lower() == 'atnum':
                    continue
                print(f"    Filtering based on {trajprop}  ")
                fprop_path = accuracy_filtered_path / trajprop
                fprop_path.mkdir(parents=True, exist_ok=True)
                molecular_filter.accuracy_filter(
                    input_path=query_data_path,
                    output_path=fprop_path,
                    prop=trajprop,
                    pjsons=prop_calc_path,
                    iter=n,
                    threshold=0.1
                )
                # clean query directory 
                dm.clean_files(query_data_path)
                # copy filtered molecules to the query data path
                shutil.copytree(fprop_path, query_data_path, dirs_exist_ok=True)
                # clean temp accuracy filtered files .xyz
                # dm.clean_files(fprop_path, '.xyz')
    else:
        skip_step_atstart.discard('acc_filter')
        

    # Step 8: Create HDF5
    print("  Running Create HDF5  ")
    run_to_hdf5(query_data_path, prop_calc_path, dm.hdf5_appending_flag, dm.output_hdf5, context_flag, dm.extra_prop_hdf5, dm.itr_hdf5_path, dm.itr_extra_prop_path, n) if 'hdf5' not in skip_step_atstart else skip_step_atstart.discard('hdf5')

    print(f"Iteration {n} completed.")


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
    parser.add_argument("--geom_optimizer", type=int, default='MACE')
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
