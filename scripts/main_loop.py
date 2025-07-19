import argparse
import os
import sys
import subprocess
from pathlib import Path
import time
import socket
from data_manage import DataManager
from parallelize import parallelize



def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    # Read time and hostname for logging
    start_loop = time.time()
    hostname = socket.gethostname()
    print(f"Starting iterating on {hostname}")

    # Number of iterations and max parallel tasks
    num_iterations = args.num_iterations
    max_parallel_tasks = args.max_parallel_tasks

    # data and code directories 
    base_dataset = args.base_dataset
    dataset_name = args.dataset_name
    blocks_path =Path(args.pipeline_blocks)
    sys.path.insert(0,str(blocks_path))
    from filter_molecules import run_filter_molecules
    from to_hdf5 import run_to_hdf5


    # Diffusion model directories
    model_code_path = Path(args.model_code)
    model_config_path = Path(args.config, "diffusion_aqm.yaml")
    
    # Define FIH-aims executable
    # ctr_fihaims_path = Path(args.config, "nctr1.in")
    fih_aims_exc = "/software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x > aims.out"

    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    # Set diffusion data and log directories in the ymal configuration file 
    dm.edit_config(model_config_path, "dataset_root", str(dm.diffusion_data_dir))
    dm.edit_config(model_config_path, "save_dir", str(dm.diffusion_log_dir))

    # Set the current hdf5 data for the loop
    hdf5_file_path = dm.input_hdf5

    # Run active learning loop
    for n in range(num_iterations):
        
        print(f"\n Iteration {n}")

        # Create directories for the current iteration
        dm.create_iteration_directories(n)

        # Step 1: Run data split
        # print(" Running Data split ")
        # script_path = model_code_path / "e3mol_aqm" / "experiments" / "data" / "aqm" / "split_data.py"
        # cmd = ["python", str(script_path), 
        #        "--file-path", str(hdf5_file_path), 
        #        "--out-path", str(dm.diffusion_data_dir)]
        # subprocess.run(cmd, check=True)

        # # step 2: Diffusion model training
        # print("  Running Diffusion model training ")
        # script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_train.py"
        # itr_model_path = dm.diffusion_model_dir / f"model_iter_{n}"
        # itr_model_path.mkdir(parents=True, exist_ok=True)
        # cmd = ["python", str(script_path),
        #        "--conf", str(model_config_path), 
        #        "--save-dir", str(itr_model_path),
        #        "--dataset", dataset_name, 
        #        "--dataset-root", str(dm.diffusion_data_dir),
        #        "--gpus", "2", "--batch-size", "32"]
        # subprocess.run(cmd, check=True)
        # model_path = itr_model_path / "best_mol_stab.ckpt"
        # # Update or just delete the model path in the config file for the next iteration
        # update_flag = n < (num_iterations - 1)
        # dm.edit_config(model_config_path, "load_ckpt_from_pretrained", str(model_path), update_flag)
        
        # # Step 3: Generate molecules
        # print("  Running Generate molecules  ")
        # script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_evaluation_qm.py"
        generated_data_path = dm.current_data / "generated"
        # n_nodes = "40"                # add function to updaten_node, embd and hlgap to target different set of molecules
        # embd = "-0.9"
        # hlgap = "-4"
        # cmd = ["python", str(script_path),
        #        "--model-path", str(model_path),
        #        "--save-dir", str(generated_data_path),
        #         "--save-xyz",
        #         "--calculate-relax-change",
        #         "--calculate-props",
        #         "--batch-size", "20",
        #         "--ngraphs", "200",
        #         "--generate-ngraphs-valid",
        #         "--num-nodes", n_nodes,
        #         "--fixed-context", embd, hlgap]
        # subprocess.run(cmd, check=True)

        # Step 4: Relax molecules
        # print("  Running Relax molecules  ")
        # script_path = blocks_path / "relax_molecules.py"
        relaxed_data_path = dm.current_data / "relaxed"
        # parallelize(script_path, generated_data_path, relaxed_data_path, max_parallel_tasks)
        
        # Step 5: Calculate SAScore and Filter molecules with low SAScore
        # print("  Running Filter molecules  ")
        # script_path = blocks_path / "calculate_sascore.py"
        filtered_data_path = dm.current_data / "filtered"
        # parallelize(script_path, relaxed_data_path, relaxed_data_path, max_parallel_tasks)
        # # Filter molecules based on SAScore
        # run_filter_molecules(relaxed_data_path, filtered_data_path)

        
        # Step 6: Calculate properties and convert output to JSON
        # print("  Running Calculate properties ")
        # script_path = blocks_path / "propcalc_preprocess.py"
        # parallelize(script_path, filtered_data_path, filtered_data_path, max_parallel_tasks)
        # parallelize(fih_aims_exc, filtered_data_path, filtered_data_path, max_parallel_tasks)
        # script_path = blocks_path / "propcalc_postprocess.py"
        # parallelize(script_path, filtered_data_path, filtered_data_path, max_parallel_tasks)


        # Step 7: Create HDF5
        print("  Running Create HDF5  ")
        run_to_hdf5(filtered_data_path, dm.hdf5_appending_flag, dm.output_hdf5, dm.extra_prop_hdf5, dm.itr_hdf5_path, dm.itr_extra_prop_path, n)
        
        # Set the loop dataset input 
        # hdf5_file_path = dm.output_hdf5
        
        # #step 8: Compress iteration files and clean up
        # print("  Running Compress and clean up  ")
        # cmd = ["tar", "-czf", f"{os.environ['DATA']}/iter_{n}.tar.gz", "-C", str(dm.current_data), "."]
        # subprocess.run(cmd, check=True)
        # cmd = ["rm", "-rf", str(dm.current_data)]
        # subprocess.run(cmd, check=True)
        # # Compress the model
        # cmd = ["tar", "-czf", f"{itr_model_path}.tar.gz", "-C", itr_model_path, "."]
        # subprocess.run(cmd, check=True)
        # cmd = ["rm", "-rf", itr_model_path]
        # subprocess.run(cmd, check=True)
        # # Clean content of data and logs in diffusion working directory
        # cmd = ["rm", "-rf", str(dm.diffusion_data_dir), str(dm.diffusion_log_dir)]
       
        print(f"Iteration {n} completed.")

    end_loop = time.time()
    print(f"Finished workflow on {hostname} Total main loop time: {end_loop - start_loop:.2f} seconds")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(end_loop - start_loop))}")

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
    parser.add_argument("--config", type=str)
    parser.add_argument("--num_iterations", type=int, default=10)

    max_parallel_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    parser.add_argument("--max_parallel_tasks", type=int, default=max_parallel_tasks)
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

#     num_threads = int(os.environ.get("SLURM_CPUS_ON_NODE", "1"))
#     env = os.environ.copy()
#     env["OMP_NUM_THREADS"] = str(num_threads)
#     env["MKL_NUM_THREADS"] = str(num_threads)
#     env["OPENBLAS_NUM_THREADS"] = str(num_threads)
#     env["NUMEXPR_NUM_THREADS"] = str(num_threads)
#     env["TF_NUM_INTEROP_THREADS"] = str(num_threads)
#     env["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
#     os.environ.update(env)

    main(args)
 
