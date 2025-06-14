import argparse
import os
import subprocess
import time
import socket
from data_manage import DataManager


def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    # import necessary functions from the pipeline
    sys.path.append(os.environ["PIPELINE"])
    from diffusion import split, train, evaluate
    from filter_molecules import apply_filter
    from to_hdf5 import create_hdf5

    # Number of iterations and max parallel tasks
    base_dataset = args.base_dataset
    global NUM_ITERATIONS, max_parallel_tasks
    NUM_ITERATIONS = args.num_iterations
    max_parallel_tasks = args.max_parallel_tasks


    start_loop = time.time()
    hostname = socket.gethostname()
    print(f"Starting iterations  on {hostname}")

    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, os.environ["DATA"])
    dm.initialize_data_directory()

    # Create active learning loop
    for n in range(NUM_ITERATIONS):
        print(f"\n Iteration {n}")
        # Create directories 
        dm.create_iteration_directories(n)

        # Step 1: Run data split
        print(" Running Data split ")
        hdf5_file = os.path.join(dm.current_data, "hdf5")
        splited_data = os.path.join(dm.current_data, "splited")
        split(hdf5_file, splited_data)

        # step 2: Diffusion model training
        print("  Running Diffusion model training ")
        model_dir = os.path.join(os.environ["MODEL_REGISTRY"], f"model_iter_{n}")
        train(splited_data, model_dir)

        # Step 3: Generate molecules
        print("  Running Generate molecules  ")
        generated_data = os.path.join(dm.current_data, "generated")
        evaluate(model_dir, generated_data) 

        # Step 4: Relax molecules
        print("  Running Relax molecules  ")
        relaxed_data = os.path.join(dm.current_data, "relaxed")
        parallelize("relax_molecules", generated_data, relaxed_data, n_jobs=max_parallel_tasks)
        
        # Step 5: Filter molecules
        print("  Running Filter molecules  ")
        filtered_data = os.path.join(dm.current_data, "filtered")
        parallelize("calculate_sascore", relaxed_data, filtered_data, n_jobs=max_parallel_tasks)
        apply_filter(relaxed_data, filtered_data)

        # Step 6: Calculate properties
        print("  Running Calculate properties ")
        properties_data = os.path.join(dm.current_data, "properties")
        parallelize("calculate_properties", filtered_data, properties_data, n_jobs=max_parallel_tasks)
        
        # Step 7: Convert to JSON
        print("  Running Convert to JSON ")
        json_data = os.path.join(dm.current_data, "properties")
        parallelize("convert_to_json", properties_data, json_data, n_jobs=max_parallel_tasks)

        # Step 8: Create HDF5
        print("  Running Create HDF5  ")
        next_hdf5_file = os.path.join(dm.next_data, "hdf5")
        create_hdf5(filtered_data, json_data, next_hdf5_file)

        print(f" Iteration {n} done")

    end_loop = time.time()
    print(f"Finished workflow on {hostname} Total main loop time: {end_loop - start_loop:.2f} seconds")


if __name__ == "__main__":
    '''
    Main entry point for the script.
    Parses command line arguments and starts the main loop.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dataset", type=str)
    parser.add_argument("--root", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--model_registry", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--pipeline", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--num_iterations", type=int, default=10)
    max_parallel_tasks = int(os.environ.get("SLURM_NTASKS", 8))
    parser.add_argument("--max_parallel_tasks", type=int, default=max_parallel_tasks)
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["ROOT"] = args.root
    os.environ["DATA"] = args.data
    os.environ["MODEL_REGISTRY"] = args.model_registry
    os.environ["MODEL_CODE"] = args.model_code
    os.environ["PIPELINE"] = args.pipeline
    os.environ["CONFIG"] = args.config

    main(args)
 