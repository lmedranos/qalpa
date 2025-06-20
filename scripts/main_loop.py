import argparse
import os
import sys
import subprocess
import time
import socket
from data_manage import DataManager
from parallelize import parallelize


def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    # import necessary functions from the pipeline
    sys.path.append(os.environ["PIPELINE"])
    from to_hdf5 import create_hdf5

    # Number of iterations and max parallel tasks
    base_dataset = args.base_dataset
    dataset_name = args.dataset_name
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
        cmd = [
            "python",
            os.path.join(os.environ["MODEL_CODE"],"e3mol_aqm/experiments/data/aqm/split_data.py"),
            "--file-path", hdf5_file,
            "--out-path", splited_data
            ]
        subprocess.run(cmd, check=True)


        # step 2: Diffusion model training
        print("  Running Diffusion model training ")
        model_dir = os.path.join(os.environ["MODEL_REGISTRY"], f"model_iter_{n}")
        cmd = [
            "python",
            os.path.join(os.environ["MODEL_CODE"], "e3mol_aqm/experiments/run_train.py"),
            "--conf", os.path.join(os.environ["CONFIG"], "diffusion_aqm.yaml"),
            "--save-dir", model_dir,
            "--dataset", dataset_name,
            "--dataset-root", splited_data,
            "--gpus", "2",
            "--batch-size", "32"
        ]
        subprocess.run(cmd, check=True)

        # Step 3: Generate molecules
        print("  Running Generate molecules  ")
        generated_data = os.path.join(dm.current_data, "generated")
        n_nodes = 24                # add function to updaten_node, embd and hlgap to target different set of molecules
        embd = -1.2
        hlgap = 6
        cmd = [
            "python",
            os.path.join(os.environ["MODEL_CODE"], "e3mol_aqm/experiments/run_evaluation_qm.py"),
            "--model-path", os.path.join(model_dir, "best_mol_stab.ckpt"),
            "--save-dir", generated_data,
            "--save-xyz",
            "--calculate-relax-change",
            "--calculate-props",
            "--batch-size", "20",
            "--ngraphs", "200",
            "--generate-ngraphs-valid",
            "--num-nodes", n_nodes,
            "--fixed-context", embd, hlgap
        ]
        subprocess.run(cmd, check=True)

        # Step 4: Relax molecules
        print("  Running Relax molecules  ")
        relaxed_data = os.path.join(dm.current_data, "relaxed")
        parallelize("relax_molecules", generated_data, relaxed_data, n_jobs=max_parallel_tasks)
        
        # Step 5: Calculate SAScore and Filter molecules with low SAScore
        print("  Running Filter molecules  ")
        filtered_data = os.path.join(dm.current_data, "filtered")
        parallelize("calculate_sascore", relaxed_data, relaxed_data, n_jobs=max_parallel_tasks)
        cmd = [
            "python",
            os.path.join(os.environ["PIPELINE"], "filter_molecules.py"),
            "--file-path", relaxed_data,
            "--out-path", filtered_data
        ]
        subprocess.run(cmd, check=True)

        # Step 6: Calculate properties and convert output to JSON
        print("  Running Calculate properties ")
        control_data = os.path.join(os.environ["CONFIG"], "nctr1.in")
        parallelize("calculate_properties", filtered_data, control_data, n_jobs=max_parallel_tasks)
        
        # Step 7: Create HDF5
        print("  Running Create HDF5  ")
        next_hdf5_file = os.path.join(dm.next_data, "hdf5")
        cmd = [
            "python",
            os.path.join(os.environ["PIPELINE"], "create_hdf5.py"),
            "--in-dir", filtered_data,
            "--out-path", next_hdf5_file
        ]
        subprocess.run(cmd, check=True)

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
    parser.add_argument("--dataset_name", type=str)
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

    num_threads = int(os.environ.get("SLURM_CPUS_ON_NODE", "1"))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    env["MKL_NUM_THREADS"] = str(num_threads)
    env["OPENBLAS_NUM_THREADS"] = str(num_threads)
    env["NUMEXPR_NUM_THREADS"] = str(num_threads)
    env["TF_NUM_INTEROP_THREADS"] = str(num_threads)
    env["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ.update(env)
    main(args)
 