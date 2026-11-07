import os
import sys
from pathlib import Path
import shutil
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.components.data_manage import DataManager
from src.components.trajectory import Trajectory
from src.components.parallelize import node_parallelize

from src.pipeline_blocks.filter_molecules import MolecularFilter
from src.pipeline_blocks.calculate_sascore import sascore
from src.pipeline_blocks.propcalc_postprocess import prop_postprocess
from src.pipeline_blocks.propcalc_preprocess import preprocess_fhi
from src.pipeline_blocks.to_hdf5 import run_to_hdf5
from src.configs.config import (
    al_paths, get_al_cfg, request_stop, resolve_blocks, resolve_start, resolve_tolerances,
    should_stop, validate_al_cfg,
)

# Allow experiment_config.yaml to derive values such as num_context_features from list lengths
OmegaConf.register_new_resolver("len", len, replace=True)

# Resolve the repository root based on this file's location so that paths in
# global_config.yaml work regardless of the current working directory.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
OmegaConf.register_new_resolver("repo_root", lambda: _REPO_ROOT, replace=True)

@hydra_main(version_base=None, config_path="../../configs", config_name="experiment_config")
def main(cfg: DictConfig):
    '''
    Main function to run the workflow.
    It sets up the environment, runs pipeline units in a loop.
    '''
    if should_stop(cfg):
        print("loop.stop is set; nothing to do.")
        return
    al = get_al_cfg(cfg)
    al_on = al["enabled"]
    al_files = al_paths(cfg)
    al_tolerances = resolve_tolerances(al, list(cfg.query.props), float(cfg.query.filter_threshold)) if al_on else None

    # Create exterapolating trajectory for properties and scores (experiment_config.yaml -> query.*)
    my_trajectory = Trajectory(cfg.query.trajectory_csv, prop_names=list(cfg.query.props), score_names=list(cfg.query.scores))
    context_flag = my_trajectory.has_properties
    if al_on:
        validate_al_cfg(al, int(cfg.EQdiffParam.num_samples), list(cfg.query.props), dataset=al["dataset"])
        if not context_flag:
            raise ValueError("Active learning needs property targets in the trajectory (query.props).")
        if not cfg.hdf5.appending_flag:
            raise ValueError("Active learning needs hdf5.appending_flag: true so that labels accumulate across iterations.")

    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory, dataset=al["dataset"])
    filteration_scheme = cfg.query.filter_scheme
              
    # Flag to resume in the midde of the loop
    pipeline_blocks = resolve_blocks(cfg)
    start_at = resolve_start(cfg, pipeline_blocks)
    if start_at in pipeline_blocks:
        active_blocks = set(pipeline_blocks[pipeline_blocks.index(start_at):])
    else:
        active_blocks = set(pipeline_blocks)
    
    # Create directories 
    dm = DataManager(cfg.base_dataset_path, cfg.data.exp_data_path)
    dm.initialize_data_directory()
    n = cfg.loop.current_iter
    if n > cfg.loop.num_iterations:
        raise ValueError(f"Current iteration {n} exceeds the total number of iterations {cfg.loop.num_iterations}.")
         
    
    dm.create_iteration_directories(n)

    prop_calc_path = dm.current_data / "prop_calc"
    # The trajectory may be shorter than the number of iterations when planning targets with active learning
    score_idx = min(n, len(my_trajectory) - 1) if al_on else n
    
    # Step 4: Calculate Score and Filter molecules 
    if 'score_filter' in active_blocks:
        if 'score' in filteration_scheme and my_trajectory.has_scores:
            print("  Running Filter molecules based on Score  ")
            score_filtered_path = dm.current_data / "score_filtered"
            for score_name in my_trajectory.score_names:
                print(f"    Filtering based on {score_name}  ")
                for f in dm.query_data.iterdir():
                    if f.is_file() and f.suffix == '.xyz' and score_name.lower() == 'sascore':
                        sascore(f, score_filtered_path)
                # Filter molecules based on SAScore
                fscored_path = score_filtered_path / score_name
                fscored_path.mkdir(parents=True, exist_ok=True)
                molecular_filter.score_filter(dm.query_data, fscored_path, score_name, score_filtered_path / f'{score_name.lower()}.txt', score_idx)
                # clean query directory
                dm.clean_dir(dm.query_data)
                # copy filtered molecules to the query data path
                shutil.copytree(fscored_path, dm.query_data, dirs_exist_ok=True)
                # clean temp score filtered files .xyz
                dm.clean_dir(fscored_path, '.xyz')
  

    # Step 4b (active learning): keep only the most informative candidates before the expensive steps
    if al_on and 'acquire' in active_blocks:
        print("  Running Acquire candidates  ")
        from src.pipeline_blocks.acquire_candidates import run_acquire
        run_acquire(al=al, prop_names=list(my_trajectory.prop_names), query_dir=dm.query_data, iter_dir=dm.current_data,
                    hdf5_path=dm.output_hdf5, cache_dir=al_files["cache_dir"], tolerances=al_tolerances,
                    seed=int(cfg.system.seed), iteration=n)

    # Step 5: Relax molecules
    if 'relax_geo' in active_blocks:
        opt_calc = cfg.geometry.optimizer
        opt_calc_model = cfg.geometry.model if cfg.geometry.model is not None else ""
        if opt_calc in ["FHI-aims", "EquiDTB", "DFTB+"]:
            print("  Running Relax molecules  ")
            relaxed_data_path = dm.current_data  / "relaxed"
            aux_arg = ["relaxed_forces", "unrelaxed_forces"] if cfg.geometry.calculate_forces else []
            kfiles = node_parallelize(cfg.pipeline_blocks.GeoRelax, dm.query_data, relaxed_data_path, aux_arg=[str(opt_calc), str(opt_calc_model), *aux_arg, "keep_temp", "cpu"], max_parallel_tasks=cfg.max_parallel_tasks, timeout_hours=cfg.processing_timeout_hours)   
            print(f"killed files during geometry optimization: {kfiles} ")  
            dm.move_kfiles(kfiles, relaxed_data_path)
            # clean query directory
            dm.clean_dir(dm.query_data)
            # Copy all relaxed molecules to the query data path
            shutil.copytree(relaxed_data_path, dm.query_data, dirs_exist_ok=True, ignore=dm.ignore_dirs)
            

    # Step 6: Calculate properties 
    if context_flag and ('prop_calc' in active_blocks):
        prop_calc = cfg.properties.calculator
        prop_calc_model = cfg.properties.model if cfg.properties.model is not None else ""
        if (prop_calc in ["FHI-aims", "EquiDTB", "DFTB+"]):
            print("  Running Calculate properties  ")
            print(f"  Running Calculate properties preprocessing using {prop_calc} ")
            if prop_calc == "FHI-aims":
                [preprocess_fhi(f, prop_calc_path, Path(cfg.code_paths.configs) / "nctr1.in") for f in dm.query_data.iterdir()]
                kfiles = node_parallelize(cfg.system.fih_aims_exc, prop_calc_path, 'aims.out', cfg.max_parallel_tasks, timeout_hours=cfg.processing_timeout_hours)
            elif prop_calc == "EquiDTB":
                kfiles = node_parallelize(cfg.pipeline_blocks.PropCalc, dm.query_data, prop_calc_path, cfg.max_parallel_tasks, aux_arg=[str(prop_calc), str(prop_calc_model), "cpu"], timeout_hours=cfg.processing_timeout_hours)                  
            else:
                kfiles = node_parallelize(cfg.pipeline_blocks.PropCalc, dm.query_data, prop_calc_path, cfg.max_parallel_tasks, aux_arg=[str(prop_calc), str(prop_calc_model), "cpu"], timeout_hours=cfg.processing_timeout_hours)
            print(f"  Killed files during property calculation: {kfiles} ")
            dm.move_kfiles(kfiles, prop_calc_path)
            # Postprocess  to create json files
            [prop_postprocess(f, prop_calc_path, prop_calc) for f in dm.query_data.iterdir() if f.is_file() and f.name not in kfiles]
        else:
            print(f"Property calculator is not supported. Skipping property calculation step.")   
            
        
    # Step 7: Filter molecules based on properties
    if 'prop_filter' in active_blocks and context_flag:
        if 'accuracy' in filteration_scheme and my_trajectory.has_properties:
            print("  Running Filter molecules based on properties  ")
            accuracy_filtered_path = dm.current_data / "accuracy_filtered"
            for trajprop in my_trajectory.prop_names:
                if trajprop.lower() == 'atnum':
                    continue
                print(f"    Filtering based on {trajprop}  ")
                fprop_path = accuracy_filtered_path / trajprop
                fprop_path.mkdir(parents=True, exist_ok=True)
                # for f in dm.query_data.iterdir():
                #             if f.is_file() and f.suffix == '.xyz':  
                #                 molecular_filter.accuracy_calculation(f, accuracy_filtered_path, trajprop, prop_calc_path, n)           
                molecular_filter.accuracy_filter(input_path=dm.query_data, output_path=fprop_path, prop=trajprop, pjsons=prop_calc_path, target_indx=n, threshold=cfg.query.filter_threshold)
                # clean query directory 
                dm.clean_dir(dm.query_data)
                # copy filtered molecules to the query data path
                shutil.copytree(fprop_path, dm.query_data, dirs_exist_ok=True)
                # clean temp accuracy filtered files .xyz
                # dm.clean_dir(fprop_path, '.xyz')

        
    # Step 8: Create HDF5
    if 'hdf5' in active_blocks:
        print("  Running Create HDF5  ")
        al_meta, al_props = None, None
        if al_on:
            from src.pipeline_blocks.prepare_hdf5_labels import prepare_al_labels
            al_props = list(my_trajectory.prop_names)
            al_meta = prepare_al_labels(query_dir=dm.query_data, prop_dir=prop_calc_path, iter_dir=dm.current_data,
                                        prop_names=al_props, tolerances=al_tolerances, iteration=n,
                                        keep_off_target=bool(al["keep_off_target"]), dataset=al["dataset"])
        run_to_hdf5(dm.query_data, prop_calc_path, cfg.hdf5.appending_flag, dm.output_hdf5, context_flag,
                    cfg.hdf5.extra_properties_flag, cfg.hdf5.hdf5_iter_flag, dm.itr_extra_prop_path, n,
                    al_meta=al_meta, al_props=al_props, dataset=al["dataset"])

    # Step 9 (active learning): metrics and early stopping
    if al_on and 'evaluate' in active_blocks:
        print("  Running Evaluate  ")
        from src.pipeline_blocks.evaluate_active_learning import run_evaluate
        _, stop, reason = run_evaluate(traj=my_trajectory, al=al, hdf5_path=dm.output_hdf5, iter_dir=dm.current_data,
                                       iteration=n, seed=int(cfg.system.seed), metrics_file=al_files["metrics_file"])
        if stop:
            print(f"  Early stop: {reason}")
            request_stop(cfg)
    
    print(f"\n Completed iteration {n} \n")
    # increment cfg.loop.current_iter and save the updated configuration to the experiment_config.yaml file
    cfg.loop.current_iter += 1
    OmegaConf.save(cfg, Path(cfg.exp_config_path))


if __name__ == "__main__":

    main()
 
