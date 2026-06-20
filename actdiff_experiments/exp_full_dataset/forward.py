import argparse
import os
import sys
import subprocess
from pathlib import Path
import shutil


def main(args):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    
    # --------------------------------------------------------------------------------
    #                               -- Intializations --
    # --------------------------------------------------------------------------------
    
    # Setup environment for GNN
    gpus_per_task = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
 
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
    from optimize_geometry import relax_molecule
    from filter_molecules import MolecularFilter
    from calculate_sascore import sascore
    from propcalc_postprocess import postprocess_fhi
    from propcalc_preprocess import preprocess_fhi
    from to_hdf5 import run_to_hdf5
    
    # Diffusion model directories
    model_code_path = Path(args.model_code)
    # sys.path.insert(0,str(model_code_path / "e3mol_aqm" / "experiments" / "data"))
    # sys.path.insert(0,str(model_code_path / "e3mol_aqm"))
    model_config_path = Path(args.config, "diffusion_aqm.yaml")
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, args.data)
    dm.initialize_data_directory()
    if dm.hdf5_appending_flag:
        hdf5_file_path = dm.output_hdf5
    else:
        hdf5_file_path = dm.input_hdf5 
        
    # Create exterapolating trajectory for properties and scores
    my_trajectory = Trajectory(args.config, prop_names=args.props, score_names=args.scores) 
    context_flag = True if my_trajectory.has_properties else False
    
    # Select geometry optimizer
    optimize_calc = args.geom_optimizer
    if optimize_calc=="SO3LR":
        opt_model = Path(os.environ["CONFIG"], "SO3LR_model.pt")
    elif optimize_calc=="MACE":
        opt_model = Path(os.environ["CONFIG"], "MACE_model_swa.model")
    elif optimize_calc=="DFTB+":
        opt_calc = optimize_calc
        opt_model = Path(os.environ["CONFIG"], "SKfiles/3ob-3-1")  # Path to Slater-Koster files for DFTB+
    else:
        opt_model = None
    
    
    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = args.filter_scheme
              
    # Flag to resume in the midde of the loop
    steps={'split', 'train', 'generate', 'relax', 'score_filter', 'pre_fih_aims', 'fih_aims', 'post_fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}

    
    # Run active learning loop
    num_iterations = args.num_iterations
    n = args.iter
    
    # --------------------------------------------------------------------------------
    #                       -- Finishing previous iteration n-1 -- 
    # --------------------------------------------------------------------------------
    
    # Redefine directories for the iteration n-1
    dm.create_iteration_directories(n-1)
    # query_data_path = dm.current_data / "query"
    print(f"Continuing iteration: {n-1}")
    
    # Step 6: Calculate properties 
    prop_calc_path = dm.current_data / "prop_calc"
    if context_flag and 'post_fih_aims' not in skip_step_atstart:
        # Postprocess FIH-aims output to create json files
        for f in dm.query_data.iterdir():
            postprocess_fhi(f, prop_calc_path)
    elif context_flag:
        skip_step_atstart.discard('post_fih_aims')
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
                for f in dm.query_data.iterdir():
                            if f.is_file() and f.suffix == '.xyz': 
                                molecular_filter.accuracy_calculation(f, accuracy_filtered_path, trajprop, prop_calc_path, n-1)           
                molecular_filter.accuracy_filter(input_path=dm.query_data, output_path=fprop_path, prop=trajprop, pjsons=prop_calc_path, target_indx=n-1, threshold=0.1)
                # clean query directory 
                dm.clean_dir(dm.query_data)
                # copy filtered molecules to the query data path
                shutil.copytree(fprop_path, dm.query_data, dirs_exist_ok=True)
                # clean temp accuracy filtered files .xyz
                dm.clean_dir(fprop_path, '.xyz')
    else:
        skip_step_atstart.discard('acc_filter')
        
    # Step 8: Create HDF5
    print("  Running Create HDF5  ")
    if 'hdf5' not in skip_step_atstart:
        run_to_hdf5(dm.query_data, prop_calc_path, dm.hdf5_appending_flag, dm.output_hdf5, context_flag, dm.extra_prop_hdf5, dm.itr_hdf5_path, dm.itr_extra_prop_path, n-1) if 'hdf5' not in skip_step_atstart else skip_step_atstart.discard('hdf5')
    else:
        skip_step_atstart.discard('hdf5')
    print(f"\n Completed iteration {n-1} \n")
    
    # --------------------------------------------------------------------------------
    #                       -- Starting new iteration --
    # --------------------------------------------------------------------------------
    
    print(f"Start new iteration: {n}")
    # Create directories for the current iteration
    dm.create_iteration_directories(n)
    # query_data_path = dm.current_data / "query"

    # Step 1: Run data split
    if 'split' not in skip_step_atstart:
        script_path = model_code_path / "e3mol_aqm" / "experiments" / "data" / "aqm" / "split_data.py"
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
    itr_model_path = dm.diffusion_model_dir / f"model_iter_{n}"
    itr_model_path.mkdir(parents=True, exist_ok=True)
    model_path = itr_model_path / "best_mol_stab.ckpt"
    if 'train' not in skip_step_atstart:
        script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_train.py"
        print("  Running Diffusion model training ")
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
            "--exclusive", "--export=ALL",
            "python", str(script_path),
            "--conf", str(model_config_path), 
            "--save-dir", str(itr_model_path),
            "--dataset", dataset_name, 
            "--dataset-root", str(dm.diffusion_data_dir),
            "--gpus", str(gpus_per_task),
            "--batch-size", "32"]
        subprocess.run(cmd, check=True, env=os.environ.copy()) 
        # Update or just delete the model path in the config file for the next iteration
        update_flag = n < (num_iterations - 1)
        dm.edit_config(model_config_path, "load_ckpt_from_pretrained", str(model_path), update_flag)
    else:
        skip_step_atstart.discard('train')
        
    # Step 3: Generate molecules
    generated_data_path = dm.current_data / "generated" 
    if 'generate' not in skip_step_atstart:
        script_path = model_code_path / "e3mol_aqm" / "experiments" / "run_evaluation_qm.py"
        print("  Running Generate molecules  ")
        # Find Target for the current iteration
        if my_trajectory.has_properties:
            target = my_trajectory.get_target(n)
        else:
            n_nodes = args.n_nodes[n]  if len(args.n_nodes) > 1 else args.n_nodes[0]
            target = ["--num-nodes", str(n_nodes)]           
        cmd = ["srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}", f"--gres=gpu:{gpus_per_task}",
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
        # clean temp .ckpt files
        dm.clean_dir(itr_model_path / "run0", ".ckpt")
        dm.clean_dir(dm.diffusion_data_dir/'raw')
        dm.clean_dir(dm.diffusion_data_dir/'processed')
        # copy generated molecules to the query data path
        shutil.copytree(generated_data_path, dm.query_data, dirs_exist_ok=True)
        #delete all files except .xyz
        dm.clean_dir(dm.query_data, exclude_suffix='.xyz')        
        # clean temp generated files .xyz
        dm.clean_dir(generated_data_path, '.xyz')
    else:
        skip_step_atstart.discard('generate')

    # Step 4: Relax molecules
    if 'relax' not in skip_step_atstart:
        if opt_model is not None and optimize_calc is not None:
            print("  Running Relax molecules  ")
            relaxed_data_path = dm.current_data  / "relaxed"
            for f in dm.query_data.iterdir():
                if f.is_file() and f.suffix == '.xyz':
                    relax_molecule(f, relaxed_data_path, optimize_calc, opt_model)           
            # clean query directory
            dm.clean_dir(dm.query_data)
            # Copy all relaxed molecules to the query data path
            shutil.copytree(relaxed_data_path, dm.query_data, dirs_exist_ok=True)
            # clean relaxed directory
            dm.clean_dir(relaxed_data_path, '.xyz')
    else:
        skip_step_atstart.discard('relax')
    
   
    # Step 5: Calculate Score and Filter molecules 
    if 'score_filter' not in skip_step_atstart:
        if 'score' in filteration_scheme and my_trajectory.has_scores:
            print("  Running Filter molecules based on Score  ")
            score_filtered_path = dm.current_data / "score_filtered"
            for score_name in my_trajectory.score_names:
                print(f"    Filtering based on {score_name}  ")
                for f in dm.query_data.iterdir():
                    if f.is_file() and f.suffix == '.xyz':
                        if score_name.lower() == 'sascore':
                            sascore(f, score_filtered_path)
                        # elif score_name.lower() == 'qed':  # To be implemented
                # Filter molecules based on SAScore
                fscored_path = score_filtered_path / score_name
                fscored_path.mkdir(parents=True, exist_ok=True)
                molecular_filter.score_filter(
                    input_path=dm.query_data,
                    output_path=fscored_path,
                    score_name=score_name,
                    score_file= score_filtered_path / f'{score_name.lower()}.txt',
                    target_indx=n
                )
                # clean query directory
                dm.clean_dir(dm.query_data)
                # copy filtered molecules to the query data path
                shutil.copytree(fscored_path, dm.query_data, dirs_exist_ok=True)
                # clean temp score filtered files .xyz
                dm.clean_dir(fscored_path, '.xyz')
    else:
        skip_step_atstart.discard('score_filter')


    # Step 6: Calculate properties 
    prop_calc_path = dm.current_data / "prop_calc"
    if context_flag and 'pre_fih_aims' not in skip_step_atstart:
        print("  Running Calculate properties preprocessing ")
        # Prepare input files for FIH-aims
        for f in dm.query_data.iterdir():
            preprocess_fhi(f, prop_calc_path, Path(os.environ["CONFIG"],"nctr1.in"))
    elif context_flag:
        skip_step_atstart.discard('pre_fih_aims')
    else:
        print("Calculating properties preprocessing is skipped.")
            
 
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

    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
