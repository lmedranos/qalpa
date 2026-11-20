import os
import sys
from pathlib import Path
import shutil
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.utils.data_manage import DataManager
from src.utils.trajectory import Trajectory
from src.utils.parallelize import node_parallelize

from src.pipeline_blocks.filter_molecules import MolecularFilter
from src.pipeline_blocks.calculate_sascore import sascore
from src.pipeline_blocks.propcalc_postprocess import prop_postprocess
from src.pipeline_blocks.propcalc_preprocess import preprocess_fhi
from src.pipeline_blocks.to_hdf5 import run_to_hdf5
from src.pipeline_blocks.score_screen import (REFERENCE_INITIAL, initial_reference, score_reference_mode,
                                              screen_query, trajectory_score_names)
from src.utils.al_config import al_paths, get_al_cfg, resolve_blocks, resolve_start, resolve_tolerances, validate_al_cfg

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
    # Create exterapolating trajectory for properties and scores (experiment_config.yaml -> query.*)
    my_trajectory = Trajectory(cfg.query.trajectory_csv, prop_names=list(cfg.query.props), score_names=trajectory_score_names(cfg))
    context_flag = my_trajectory.has_properties

    # Optional active-learning target planning (experiment_config.yaml -> al.*, defaults in src/configs/al.yaml)
    al = get_al_cfg(cfg)
    al_on = bool(al["enabled"])
    validate_al_cfg(cfg, al, my_trajectory)

    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    filteration_scheme = cfg.query.filter_scheme
              
    # Flag to resume in the midde of the loop
    pipeline_blocks = resolve_blocks(cfg, al)
    start_at = resolve_start(cfg, al, pipeline_blocks)
    if start_at in pipeline_blocks:
        active_blocks = set(pipeline_blocks[pipeline_blocks.index(start_at):])
    else:
        active_blocks = set(pipeline_blocks)
    
    # Create directories 
    dm = DataManager(cfg.base_dataset_path, cfg.data.exp_data_path)
    dm.hdf5_appending_flag = bool(cfg.hdf5.appending_flag)
    dm.hdf5_iter_flag = bool(cfg.hdf5.hdf5_iter_flag)
    dm.extra_properties_flag = bool(cfg.hdf5.extra_properties_flag)
    dm.initialize_data_directory()
    n = cfg.loop.current_iter
    if n > cfg.loop.num_iterations:
        raise ValueError(f"Current iteration {n} exceeds the total number of iterations {cfg.loop.num_iterations}.")
         
    
    dm.create_iteration_directories(n)

    prop_calc_path = dm.current_data / "prop_calc"
    # With planned targets the trajectory may be shorter than the number of iterations
    score_idx = min(n, len(my_trajectory) - 1) if al_on else n
    
    # (dedup runs at the end of the forward step, so query/ already holds only the deduplicated molecules)

    # Step 4: Calculate Score and Filter molecules 
    if 'score_filter' in active_blocks and score_reference_mode(cfg) == REFERENCE_INITIAL:
        if 'score' in filteration_scheme:
            # QED > mean QED x (1 - r_QED) and SA < mean SA x (1 + r_SA) of the initial dataset (both; query.score_relax)
            print("  Running Filter molecules based on QED and SA score (initial-dataset means)  ")
            reference = initial_reference(dm.input_hdf5)
            relax = cfg.query.get("score_relax")
            relax = OmegaConf.to_container(relax, resolve=True) if relax is not None else None
            screen_query(dm.query_data, dm.current_data / "score_filtered", reference, relax=relax,
                         scores=list(cfg.query.get("scores", ["SAScore", "QED"])))
        else:
            print("  score_filter skipped: 'score' is not in query.filter_scheme")
    elif 'score_filter' in active_blocks:
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
                # same summary file as the initial_mean screen, so the counts survive the cleanup below
                n_in = sum(1 for f in dm.query_data.iterdir() if f.is_file() and f.suffix == '.xyz')
                n_pass = sum(1 for f in fscored_path.iterdir() if f.is_file() and f.suffix == '.xyz')
                import json
                (score_filtered_path / "score_screen_summary.json").write_text(json.dumps({
                    "reference": "trajectory", "score": score_name,
                    "threshold": float(my_trajectory.scores[score_name][score_idx]),
                    "counts": {"total": n_in, "passed": n_pass, "rejected": n_in - n_pass}}, indent=2))
                # clean query directory
                dm.clean_dir(dm.query_data)
                # copy filtered molecules to the query data path
                shutil.copytree(fscored_path, dm.query_data, dirs_exist_ok=True)
                # clean temp score filtered files .xyz
                dm.clean_dir(fscored_path, '.xyz')
  

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
    if 'prop_filter' in active_blocks and context_flag and al_on:
        if 'accuracy' in filteration_scheme:
            # active learning: each molecule against the target it was generated for (anchor or additional)
            print("  Running Filter molecules based on properties (per planned target)  ")
            from src.pipeline_blocks.label_targets import filter_by_own_target
            filter_props = list(my_trajectory.plan_names)
            filter_by_own_target(query_dir=dm.query_data, prop_dir=prop_calc_path, iter_dir=dm.current_data,
                                 prop_names=filter_props, iteration=n,
                                 tolerances=resolve_tolerances(al, filter_props, float(cfg.query.filter_threshold)))
        else:
            print("  prop_filter skipped: 'accuracy' is not in query.filter_scheme")
    elif 'prop_filter' in active_blocks and context_flag:
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
            # hit/miss against each molecule's own planned target (after prop_filter, only hits remain)
            from src.pipeline_blocks.label_targets import prepare_al_labels
            al_props = list(my_trajectory.plan_names)
            al_meta = prepare_al_labels(query_dir=dm.query_data, prop_dir=prop_calc_path, iter_dir=dm.current_data,
                                        prop_names=al_props, iteration=n,
                                        tolerances=resolve_tolerances(al, al_props, float(cfg.query.filter_threshold)),
                                        keep_off_target=bool(al["keep_off_target"]))
        run_to_hdf5(
            dm.query_data,
            prop_calc_path,
            cfg.hdf5.appending_flag,
            dm.output_hdf5,
            context_flag,
            dm.extra_prop_hdf5 if cfg.hdf5.extra_properties_flag else None,
            dm.itr_hdf5_path if cfg.hdf5.hdf5_iter_flag else None,
            dm.itr_extra_prop_path if cfg.hdf5.extra_properties_flag else None,
            n,
            al_meta=al_meta,
            al_props=al_props,
        )

    # Step 9 (active learning): hit rate and corridor coverage
    if al_on and 'evaluate' in active_blocks:
        print("  Running Evaluate  ")
        from src.pipeline_blocks.evaluate_targets import run_evaluate
        run_evaluate(traj=my_trajectory, al=al, hdf5_path=dm.output_hdf5, iteration=n, seed=int(cfg.system.seed),
                     metrics_file=al_paths(cfg)["metrics_file"], iter_dir=dm.current_data)
    
    print(f"\n Completed iteration {n} \n")
    # increment cfg.loop.current_iter and save the updated configuration to the experiment_config.yaml file
    cfg.loop.current_iter += 1
    OmegaConf.save(cfg, Path(cfg.exp_config_path))


if __name__ == "__main__":

    main()
 
