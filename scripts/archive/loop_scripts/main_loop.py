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
    
    # Read time and hostname for logging
    start_loop = time.time()
    hostname = socket.gethostname()
    print(f"Starting iterating on {hostname}")

    # Number of iterations and max parallel tasks
    num_iterations = args.num_iterations
    max_parallel_tasks = args.max_parallel_tasks

    # utils directories
    utils_path =Path(args.actdiff_utils)
    sys.path.insert(0,str(utils_path))
    from data_manage import DataManager
    from parallelize import parallelize
    from trajectory import Trajectory
    
    # data directories 
    base_dataset = args.base_dataset
    dataset_name = args.dataset_name
    
    # Code directories 
    blocks_path =Path(args.pipeline_blocks)
    sys.path.insert(0,str(blocks_path))
    from filter_molecules import MolecularFilter
    from to_hdf5 import run_to_hdf5


    # Diffusion model directories
    model_code_path = Path(args.model_code)
    sys.path.insert(0,str(model_code_path / "e3mol_aqm" / "experiments" / "data"))
    sys.path.insert(0,str(model_code_path / "e3mol_aqm"))
    model_config_path = Path(args.config, "diffusion_aqm.yaml")
    
    # Define FIH-aims executable
    fih_aims_exc = " /software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x"

    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    
    # Set diffusion data and log directories in the ymal configuration file 
    dm.edit_config(model_config_path, "dataset_root", str(dm.diffusion_data_dir))
    dm.edit_config(model_config_path, "save_dir", str(dm.diffusion_log_dir))
    dm.edit_config(model_config_path, "test_save_dir", str(dm.diffusion_test_dir))

    # Set the intial hdf5 data for the loop 
    hdf5_file_path = dm.input_hdf5

    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    context_flag = True if my_trajectory.has_properties else False
    dm.edit_config(model_config_path, "context_mapping", str(context_flag))
    dm.edit_config(model_config_path, "properties_list", [p for p in my_trajectory.prop_names if p != "atNUM"])   # Exclude 'atNUM' from properties list in the config file
    dm.edit_config(model_config_path, "num_context_features", str(len(my_trajectory.prop_names)- (1 if "atNUM" in my_trajectory.prop_names else 0)))  # Exclude 'atNUM' from context features in the config file
    
    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = args.filter_scheme

    # compress flag to compress iteration files
    compress_flag = False

    # Flag to resume in the midde of the loop
    resume_itr = 1          # from 1 to num_iterations
    steps={'split', 'train', 'generate', 'relax', 'score_filter', 'fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}
    # Run active learning loop
    for n in range(num_iterations):

        # Skip iterations that are already completed
        if (n+1) < resume_itr:
            hdf5_file_path = dm.output_hdf5
            print(f"Skipping iteration {n} as it is already completed.")
            continue

        print(f"\n Iteration {n}")

        # Setup environment for GNN
        gpus_per_task = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        # env = os.environ.copy()   
        # env["OMP_NUM_THREADS"] = str(cpus_per_task)
        # env["MKL_NUM_THREADS"] = str(cpus_per_task)
        # os.environ.update(env)
        
        # Create directories for the current iteration
        dm.create_iteration_directories(n)

        # Step 1: Run data split
        script_path = model_code_path / "e3mol_aqm" / "experiments" / "data" / "aqm" / "split_data.py"
        if 'split' not in skip_step_atstart:
            print(" Running Data split ")
            cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}",
                "--exclusive", "--export=ALL",
                "python", str(script_path),
                "--file-path", str(hdf5_file_path),
                "--out-path", str(dm.diffusion_data_dir)
                ]
            subprocess.run(cmd, check=True)  
        else:
            skip_step_atstart.discard('split')
            
        # step 2: Diffusion model training
        script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_train.py"
        itr_model_path = dm.diffusion_model_dir / f"model_iter_{n}"
        itr_model_path.mkdir(parents=True, exist_ok=True)
        if 'train' not in skip_step_atstart:
            print("  Running Diffusion model training ")
            cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", "--gres=gpu:2",
                "--exclusive", "--export=ALL",
                "python", str(script_path),
                "--conf", str(model_config_path), 
                "--save-dir", str(itr_model_path),
                "--dataset", dataset_name, 
                "--dataset-root", str(dm.diffusion_data_dir),
                "--gpus", str(gpus_per_task),
                "--batch-size", "32"]
            subprocess.run(cmd, check=True, env=os.environ.copy())  
            model_path = itr_model_path / "best_mol_stab.ckpt"
            # Update or just delete the model path in the config file for the next iteration
            update_flag = n < (num_iterations - 1)
            dm.edit_config(model_config_path, "load_ckpt_from_pretrained", str(model_path), update_flag)
        else:
            skip_step_atstart.discard('train')
            
    # Step 3: Generate molecules
        script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_evaluation_qm.py"
        generated_data_path = dm.current_data / "generated" 
        if 'generate' not in skip_step_atstart:
            print("  Running Generate molecules  ")
            # Find Target for the current iteration
            if my_trajectory.has_properties:
                target = my_trajectory.get_target(n)
            else:
                n_nodes = args.n_nodes[n]  if len(args.n_nodes) > 1 else args.n_nodes[0]
                target = ["--num-nodes", str(n_nodes)]           
            cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}",
                    "--exclusive", "--export=ALL",
                    "python", str(script_path),
                    "--model-path", str(model_path),
                    "--save-dir", str(generated_data_path),
                    "--save-xyz", 
                    "--batch-size", "20",
                    "--ngraphs", "200",
                    "--generate-ngraphs-valid",
                    *target,
                    ]
            subprocess.run(cmd, check=True, env=os.environ.copy()) 
        else:
            skip_step_atstart.discard('generate')

        # ---- query path to choose molecules fot the next iteration ----
        
        # clean temp .ckpt files
        dm.clean_files(itr_model_path / "run0", ".ckpt")
        dm.clean_files(dm.diffusion_data_dir/'raw')
        dm.clean_files(dm.diffusion_data_dir/'processed')
        
        # create query directory and copy all generated molecules to the query data path
        query_data_path = dm.current_data / "query"
        shutil.copytree(generated_data_path, query_data_path, dirs_exist_ok=True)
        
        #delete all files except .xyz
        for item in query_data_path.iterdir():
            if item.is_file() and item.suffix != '.xyz':
                item.unlink()  
        # clean temp generated files .xyz
        # dm.clean_files(generated_data_path, '.xyz')


        # Step 4: Relax molecules
        script_path = blocks_path / "relax_molecules.py"
        relaxed_data_path = dm.current_data / "relaxed"
        if 'relax' not in skip_step_atstart:
            print("  Running Relax molecules  ")
            parallelize(script_path, query_data_path, relaxed_data_path, max_parallel_tasks) 
            # clean query directory
            dm.clean_files(query_data_path)
            # Copy all relaxed molecules to the query data path
            shutil.copytree(relaxed_data_path, query_data_path, dirs_exist_ok=True)
            # # clean temp relaxed files .xyz
            # dm.clean_files(relaxed_data_path, '.xyz')
        else:
            skip_step_atstart.discard('relax')


        # Step 5: Calculate Score and Filter molecules 
        if 'score_filter' not in skip_step_atstart:
            if 'score' in filteration_scheme and my_trajectory.has_scores:
                print("  Running Filter molecules based on Score  ")
                score_filtered_path = dm.current_data / "score_filtered"
                for score_name in my_trajectory.score_names:
                    print(f"    Filtering based on {score_name}  ")
                    fscored_path = score_filtered_path / score_name
                    fscored_path.mkdir(parents=True, exist_ok=True)
                    if score_name.lower() == 'sascore':
                        script_path = blocks_path / "calculate_sascore.py"
                    # elif score_name.lower() == 'qed':  # To be implemented
                        # script_path = blocks_path / "calculate_qed.py"
                    parallelize(script_path, query_data_path, score_filtered_path, max_parallel_tasks)
                    # Filter molecules based on SAScore
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
                    # clean temp score filtered files .xyz
                    # dm.clean_files(fscored_path, '.xyz')
        else:
            skip_step_atstart.discard('score_filter')



        # Step 6: Calculate properties 
        prop_calc_path = dm.current_data / "prop_calc"
        if context_flag and 'fih_aims' not in skip_step_atstart:
            print("  Running Calculate properties ")
            script_path = blocks_path / "propcalc_preprocess.py"
            parallelize(script_path, query_data_path, prop_calc_path, max_parallel_tasks) 
            parallelize(fih_aims_exc, prop_calc_path, 'aims.out', max_parallel_tasks) 
            script_path = blocks_path / "propcalc_postprocess.py"
            parallelize(script_path, prop_calc_path, prop_calc_path, max_parallel_tasks)  
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
        

        # step 9: Compress iteration files and clean up
        if compress_flag and 'compress' not in skip_step_atstart:
            print("  Running Compress and clean up  ")
            archive_path = f"{args.data}/iter_{n}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(str(dm.current_data), arcname=".")
            shutil.rmtree(dm.current_data)
            # Compress the model and clean diffusion working directories
            archive_path = itr_model_path.with_suffix('.tar.gz')
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(str(itr_model_path), arcname=".")
            shutil.rmtree(itr_model_path)
            shutil.rmtree(dm.diffusion_data_dir)
            shutil.rmtree(dm.diffusion_log_dir)
        elif "compress" in skip_step_atstart:
            skip_step_atstart.discard('compress')

        # Set the loop dataset input 
        hdf5_file_path = dm.output_hdf5
        
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
    parser.add_argument("--actdiff_utils", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--props", type=str, nargs='+', default=[])
    parser.add_argument("--scores", type=str, nargs='+', default=[])
    parser.add_argument("--n_nodes", type=int, nargs='+', default=[20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    parser.add_argument("--filter_scheme", type=str, nargs='+', default=[])
    parser.add_argument("--max_parallel_tasks", type=int, default=1)
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    # cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    # env = os.environ.copy()
    # env["OMP_NUM_THREADS"] = str(cpus_per_task)
    # env["MKL_NUM_THREADS"] = str(cpus_per_task)
    # env["OPENBLAS_NUM_THREADS"] = str(cpus_per_task)
    # env["NUMEXPR_NUM_THREADS"] = str(cpus_per_task)
 

    # os.environ.update(env)

    main(args)
 
