import argparse
import os
import sys 
from pathlib import Path
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
    
    # Code directories 
    blocks_path =Path(args.pipeline_blocks)
    sys.path.insert(0,str(blocks_path))
    from optimize_geometry import relax_molecule
    from filter_molecules import MolecularFilter
    from calculate_sascore import sascore
    from propcalc_postprocess import postprocess_fhi
    from propcalc_preprocess import preprocess_fhi

    # Define FIH-aims executable
    fih_aims_exc = " /software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x"
    
    # Create the intial hdf5 data directory if it doesn't exist
    dm = DataManager(args.base_dataset, args.data)
    dm.initialize_data_directory()
    
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
        
    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = args.filter_scheme
      
    # Number of models to evaluate
    num_models = args.num_models
    num_targets = len(my_trajectory)   # len(my_trajectory.target_values)
    
    # Flag to resume in the midde of the loop
    steps={'split', 'train', 'generate', 'relax', 'score_filter', 'pre_fih_aims', 'fih_aims', 'post_fih_aims', 'acc_filter', 'hdf5', 'compress'}
    skip_step_atstart = {}


    for m in range(num_models):    
        for t in range(num_targets): 
            dm.create_evaluation_directories(t, m)

            prop_calc_path = dm.current_data / "prop_calc"
            
            # Step 2: Relax molecules
            if 'relax' not in skip_step_atstart:
               if opt_calc is not None:
                print("  Running Relax molecules  ")
                relaxed_data_path = dm.current_data  / "relaxed"
                geo_opt_script = blocks_path / "optimize_geometry.py"
                kfiles = node_parallelize(geo_opt_script, dm.query_data, relaxed_data_path, aux_arg=[str(opt_calc), str(opt_model), "relaxed_forces", "unrelaxed_forces", "keep_temp", "cpu"])         # remove flags for forces and keep_temp if not needed
                print(f"killed files during geometry optimization: {kfiles} ")  
                kdir = relaxed_data_path / "killed"
                kdir.mkdir(parents=True, exist_ok=True)
                for kf in kfiles:
                    if kf == "unknown":
                        continue
                    src = dm.query_data / kf
                    dest = kdir / kf
                    if src.exists():
                        shutil.move(str(src), str(dest))
                # clean query directory
                dm.clean_dir(dm.query_data)
                # Copy all relaxed molecules to the query data path
                shutil.copytree(relaxed_data_path, dm.query_data, dirs_exist_ok=True, ignore=dm.ignore_dirs)
                # clean relaxed directory
                # dm.clean_dir(relaxed_data_path, '.xyz')
            else:
                skip_step_atstart.discard('relax')
                
            # Step 3: Calculate Score 
            if 'score_filter' not in skip_step_atstart:
                if 'score' in filteration_scheme and my_trajectory.has_scores:
                    print("  Running Calculating Score  ")
                    for score_name in my_trajectory.score_names:
                        print(f"    Calculation: {score_name}  ")
                        for f in dm.query_data.iterdir():
                            if f.is_file() and f.suffix == '.xyz':
                                if score_name.lower() == 'sascore':
                                    sascore(f, dm.results)
                                # elif score_name.lower() == 'qed':  # To be implemented
            else:
                skip_step_atstart.discard('score_filter')

            # Step 4: preparing for Calculating properties 
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
            

            # # Step 4: Calculate properties 
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
 
