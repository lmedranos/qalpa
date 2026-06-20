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

    # utils directories
    utils_path =Path(args.actdiff_utils)
    sys.path.insert(0,str(utils_path))
    from data_manage import DataManager
    from trajectory import Trajectory
    from parallelize import node_parallelize

    # data directories 
    base_dataset = args.base_dataset
    dataset_name = args.dataset_name
    
    # Code directories 
    blocks_path =Path(args.pipeline_blocks)
    sys.path.insert(0,str(blocks_path))
    from optimize_geometry import relax_molecule
    from filter_molecules import MolecularFilter
    from calculate_sascore import sascore
    from propcalc_postprocess import prop_postprocess
    from propcalc_preprocess import preprocess_fhi
    from to_hdf5 import run_to_hdf5
    

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
        opt_model = Path(os.environ["CONFIG"]).parent / "SO3LR_model.pt"
    elif optimize_calc=="MACE":
        opt_model = Path(os.environ["CONFIG"]).parent / "MACE_model_swa.model"
    elif optimize_calc=="DFTB+":
        opt_calc = optimize_calc
        opt_model = None
    elif optimize_calc=="EquiDTB":
        opt_calc = optimize_calc
        opt_model = Path(os.environ["CONFIG"]).parent / "EquiDTB3.model"  # Path to the MACE model trained to reproduce DFTB+ results
    else:
        opt_model = None
    
    # select property calculator
    if args.prop_calculator == "FHI-aims":
        prop_calc = "FHI-aims"
    elif args.prop_calculator == "EquiDTB":
        prop_calc = "EquiDTB"
        prop_model = Path(os.environ["CONFIG"]).parent / "EquiDTB3.model"  # Path to the MACE model trained to reproduce DFTB+ results
    else:
        prop_calc = None
    
    
    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = args.filter_scheme
              
    # Flag to resume in the midde of the loop
    steps={'split', 'train', 'generate', 'relax', 'score_filter', 'pre_fih_aims', 'fih_aims', 'post_fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}

    
    # Run active learning loop
    num_iterations = args.num_iterations
    n = args.iter
    # Number of iterations and max parallel tasks
    max_parallel_tasks = args.max_parallel_tasks
    
    forces_calculation = args.forces_calculation
    filteration_threshold = args.filteration_threshold
   
    # Create directories for the current iteration
    dm.create_iteration_directories(n)


    # Step 4: Calculate Score and Filter molecules 
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


    # Step 5: Relax molecules
    if 'relax' not in skip_step_atstart:
        if opt_calc is not None:
            print("  Running Relax molecules  ")
            relaxed_data_path = dm.current_data  / "relaxed"
            # for f in dm.query_data.iterdir():
            #     if f.is_file() and f.suffix == '.xyz':
            #         relax_molecule(f, relaxed_data_path, opt_calc, opt_model, relaxed_forces=True, unrelaxed_forces=True, keep_temp=True)           
            geo_opt_script = blocks_path / "optimize_geometry.py"
            if forces_calculation:
                aux_arg = ["relaxed_forces", "unrelaxed_forces"]
            kfiles = node_parallelize(geo_opt_script, dm.query_data, relaxed_data_path, aux_arg=[str(opt_calc), str(opt_model), *aux_arg, "keep_temp", "cpu"])         # remove flags for forces and keep_temp if not needed
            print(f"killed files during geometry optimization: {kfiles} ")  
            dm.move_kfiles(kfiles, relaxed_data_path)
            # clean query directory
            dm.clean_dir(dm.query_data)
            # Copy all relaxed molecules to the query data path
            shutil.copytree(relaxed_data_path, dm.query_data, dirs_exist_ok=True, ignore=dm.ignore_dirs)
            # clean relaxed directory
            # dm.clean_dir(relaxed_data_path, '.xyz')
    else:
        skip_step_atstart.discard('relax')
    


    # Step 6: Calculate properties 
    if context_flag:
        if ('fih_aims' not in skip_step_atstart):
            print("  Running Calculate properties  ")
            if (prop_calc in ["FHI-aims", "EquiDTB"]):
                prop_calc_path = dm.current_data / "prop_calc"
                print(f"  Running Calculate properties preprocessing using {prop_calc} ")
                if prop_calc == "FHI-aims":
                    [preprocess_fhi(f, prop_calc_path, Path(os.environ["CONFIG"],"nctr1.in")) for f in dm.query_data.iterdir()]
                    fih_aims_exc = "/software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x"
                    kfiles = node_parallelize(fih_aims_exc, prop_calc_path, 'aims.out', max_parallel_tasks) 
                elif prop_calc == "EquiDTB":
                    prop_calc_script =  blocks_path / "prop_calc.py"
                    kfiles = node_parallelize(prop_calc_script, dm.query_data, prop_calc_path, aux_arg=[str(prop_calc), str(prop_model)])         
                    
                dm.move_kfiles(kfiles, prop_calc_path)
                # Postprocess  to create json files
                [prop_postprocess(f, prop_calc_path, prop_calc) for f in dm.query_data.iterdir() if f.is_file() and f.name not in kfiles]

            else:
                print(f"Property calculator is not supported. Skipping property calculation step.")   
        else:
            print("Calculating properties is skipped for this iteration.")
            skip_step_atstart.discard('property_calculation')        
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
                                molecular_filter.accuracy_calculation(f, accuracy_filtered_path, trajprop, prop_calc_path, n)           
                molecular_filter.accuracy_filter(input_path=dm.query_data, output_path=fprop_path, prop=trajprop, pjsons=prop_calc_path, target_indx=n, threshold=filteration_threshold)
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
        run_to_hdf5(dm.query_data, prop_calc_path, dm.hdf5_appending_flag, dm.output_hdf5, context_flag, dm.extra_prop_hdf5, dm.itr_hdf5_path, dm.itr_extra_prop_path, n) if 'hdf5' not in skip_step_atstart else skip_step_atstart.discard('hdf5')
    else:
        skip_step_atstart.discard('hdf5')
    print(f"\n Completed iteration {n} \n")


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
    parser.add_argument("--forces_calculation", default=True)
    parser.add_argument("--prop_calculator", type=str, default="FHI-aims")
    parser.add_argument("--filteration_threshold", type=float, default=0.1)
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CONFIG"] = args.config

    main(args)
 
