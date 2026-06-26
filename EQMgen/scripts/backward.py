"""
EQMgen.scripts.backward - Backward pass: training on query data and evaluating properties

This module handles:
1. Score-based filtering
2. Geometry optimization/relaxation
3. Property calculations
4. Accuracy-based filtering
5. Data export to HDF5
"""

import os
import sys
from pathlib import Path
import shutil
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


def backward_main(
    config_path: str = None,
    base_dataset: str = None,
    data_path: str = None,
    pipeline_blocks: list = None,
    start_at: str = "score_filter",
    num_iterations: int = 10,
    current_iter: int = 0,
    max_parallel_tasks: int = 1,
    optimizer: str = "DFTB+",
    calculate_forces: bool = True,
    filter_scheme: list = None,
    calculator: str = "DFTB+",
    filteration_threshold: float = 0.1,
    props: list = None,
    scores: list = None,
    cfg: DictConfig = None,
):
    """
    Main function to run the backward workflow pass.
    
    Args:
        config_path: Path to config directory
        base_dataset: Path to base HDF5 dataset
        data_path: Path to data directory
        pipeline_blocks: List of pipeline blocks to execute
        start_at: Step to start from
        num_iterations: Total number of iterations
        current_iter: Current iteration number
        max_parallel_tasks: Maximum parallel tasks
        optimizer: Geometry optimizer (DFTB+, FHI-aims, EquiDTB, SO3LR, MACE)
        calculate_forces: Whether to calculate forces
        filter_scheme: Filtering scheme (score, accuracy)
        calculator: Property calculator (FHI-aims, EquiDTB)
        filteration_threshold: Threshold for accuracy filtering
        props: List of property names
        scores: List of score names
        cfg: OmegaConf DictConfig (used when called from Hydra)
    """
    # Use Hydra config if provided, otherwise use arguments
    if cfg is not None:
        os.environ["CONFIG"] = cfg.paths.config
        utils_path = Path(cfg.paths.actdiff_utils)
        config_path = cfg.paths.config
        base_dataset = cfg.data.base_dataset_path
        data_path = cfg.data.data_path
        pipeline_blocks = cfg.pipeline.blocks
        start_at = cfg.pipeline.start_at
        optimizer = cfg.geometry.optimizer
        calculate_forces = cfg.geometry.calculate_forces
        filter_scheme = cfg.geometry.filter_scheme
        calculator = cfg.properties.calculator
        filteration_threshold = cfg.properties.filteration_threshold
        num_iterations = cfg.training.num_iterations
        current_iter = cfg.training.current_iter
        max_parallel_tasks = cfg.training.max_parallel_tasks
        props = cfg.data.get("props", [])
        scores = cfg.data.get("scores", [])
        blocks_path = Path(cfg.paths.pipeline_blocks)
    else:
        os.environ["CONFIG"] = config_path
        utils_path = Path(config_path).parent / "utils"
        blocks_path = Path(config_path).parent / "pipeline_blocks"
    
    if props is None:
        props = []
    if scores is None:
        scores = []
    if pipeline_blocks is None:
        pipeline_blocks = ["score_filter", "relax_geo", "prop_calc", "prop_filter", "hdf5"]
    if filter_scheme is None:
        filter_scheme = ["score", "accuracy"]
    
    # utils directories
    sys.path.insert(0, str(utils_path))
    from data_manage import DataManager
    from trajectory import Trajectory
    from parallelize import node_parallelize

    # Code directories 
    sys.path.insert(0, str(blocks_path))
    from optimize_geometry import relax_molecule
    from filter_molecules import MolecularFilter
    from calculate_sascore import sascore
    from propcalc_postprocess import prop_postprocess
    from propcalc_preprocess import preprocess_fhi
    from to_hdf5 import run_to_hdf5
    

    # Create the initial hdf5 data directory if it doesn't exist
    dm = DataManager(base_dataset, data_path)
    dm.initialize_data_directory()
    if dm.hdf5_appending_flag:
        hdf5_file_path = dm.output_hdf5
    else:
        hdf5_file_path = dm.input_hdf5 
        
    # Create extrapolating trajectory for properties and scores
    my_trajectory = Trajectory(config_path, prop_names=props, score_names=scores) 
    context_flag = True if my_trajectory.has_properties else False
    
    # Select geometry optimizer
    if optimizer == "SO3LR":
        opt_model = Path(os.environ["CONFIG"]).parent / "SO3LR_model.pt"
        opt_calc = optimizer
    elif optimizer == "MACE":
        opt_model = Path(os.environ["CONFIG"]).parent / "MACE_model_swa.model"
        opt_calc = optimizer
    elif optimizer == "DFTB+":
        opt_calc = optimizer
        opt_model = None
    elif optimizer == "EquiDTB":
        opt_calc = optimizer
        opt_model = Path(os.environ["CONFIG"]).parent / "EquiDTB3.model"
    else:
        opt_calc = None
        opt_model = None
    
    # select property calculator
    if calculator == "FHI-aims":
        prop_calc = "FHI-aims"
        prop_model = None
    elif calculator == "EquiDTB":
        prop_calc = "EquiDTB"
        prop_model = Path(os.environ["CONFIG"]).parent / "EquiDTB3.model"
    else:
        prop_calc = None
        prop_model = None
    
    # Create Molecular Filter
    molecular_filter = MolecularFilter(my_trajectory)
    
    # Flag to resume in the middle of the loop
    skip_step_atstart = set(pipeline_blocks[pipeline_blocks.index(start_at):])
   
    # Create directories for the current iteration
    dm.create_iteration_directories(current_iter)
    
    # Step 4: Calculate Score and Filter molecules 
    if 'score_filter' in pipeline_blocks and 'score_filter' not in skip_step_atstart:
        if 'score' in filter_scheme and my_trajectory.has_scores:
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
                molecular_filter.score_filter(dm.query_data, fscored_path, score_name, score_filtered_path / f'{score_name.lower()}.txt', current_iter)
                # clean query directory
                dm.clean_dir(dm.query_data)
                # copy filtered molecules to the query data path
                shutil.copytree(fscored_path, dm.query_data, dirs_exist_ok=True)
                # clean temp score filtered files .xyz
                dm.clean_dir(fscored_path, '.xyz')
  

    # Step 5: Relax molecules
    if 'relax' in pipeline_blocks and 'relax' not in skip_step_atstart:
        if opt_calc is not None:
            print("  Running Relax molecules  ")
            relaxed_data_path = dm.current_data / "relaxed"
            geo_opt_script = blocks_path / "optimize_geometry.py"
            aux_arg = []
            if calculate_forces:
                aux_arg = ["relaxed_forces", "unrelaxed_forces"]
            kfiles = node_parallelize(
                geo_opt_script, 
                dm.query_data, 
                relaxed_data_path, 
                aux_arg=[str(opt_calc), str(opt_model), *aux_arg, "keep_temp", "cpu"]
            )
            print(f"killed files during geometry optimization: {kfiles} ")  
            dm.move_kfiles(kfiles, relaxed_data_path)
            # clean query directory
            dm.clean_dir(dm.query_data)
            # Copy all relaxed molecules to the query data path
            shutil.copytree(relaxed_data_path, dm.query_data, dirs_exist_ok=True, ignore=dm.ignore_dirs)
    
    
    # Step 6: Calculate properties 
    if context_flag and ('prop_calc' in pipeline_blocks) and ('prop_calc' not in skip_step_atstart):
        print("  Running Calculate properties  ")
        if prop_calc in ["FHI-aims", "EquiDTB"]:
            prop_calc_path = dm.current_data / "prop_calc"
            print(f"  Running Calculate properties preprocessing using {prop_calc} ")
            if prop_calc == "FHI-aims":
                [preprocess_fhi(f, prop_calc_path, Path(os.environ["CONFIG"], "nctr1.in")) for f in dm.query_data.iterdir()]
                fih_aims_exc = "/software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x"
                kfiles = node_parallelize(fih_aims_exc, prop_calc_path, 'aims.out', max_parallel_tasks) 
            elif prop_calc == "EquiDTB":
                prop_calc_script = blocks_path / "prop_calc.py"
                kfiles = node_parallelize(prop_calc_script, dm.query_data, prop_calc_path, aux_arg=[str(prop_calc), str(prop_model)])                  
            dm.move_kfiles(kfiles, prop_calc_path)
            # Postprocess to create json files
            [prop_postprocess(f, prop_calc_path, prop_calc) for f in dm.query_data.iterdir() if f.is_file() and f.name not in kfiles]
        else:
            print(f"Property calculator is not supported. Skipping property calculation step.")   
            
       
    # Step 7: Filter molecules based on properties
    if 'prop_filter' in pipeline_blocks and 'prop_filter' not in skip_step_atstart:
        if 'accuracy' in filter_scheme and my_trajectory.has_properties:
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
                        molecular_filter.accuracy_calculation(f, accuracy_filtered_path, trajprop, prop_calc_path, current_iter)           
                molecular_filter.accuracy_filter(
                    input_path=dm.query_data, 
                    output_path=fprop_path, 
                    prop=trajprop, 
                    pjsons=prop_calc_path, 
                    target_indx=current_iter, 
                    threshold=filteration_threshold
                )
                # clean query directory 
                dm.clean_dir(dm.query_data)
                # copy filtered molecules to the query data path
                shutil.copytree(fprop_path, dm.query_data, dirs_exist_ok=True)
                # clean temp accuracy filtered files .xyz
                dm.clean_dir(fprop_path, '.xyz')

        
    # Step 8: Create HDF5
    print("  Running Create HDF5  ")
    if 'hdf5' in pipeline_blocks and 'hdf5' not in skip_step_atstart:
        run_to_hdf5(
            dm.query_data, 
            prop_calc_path, 
            dm.hdf5_appending_flag, 
            dm.output_hdf5, 
            context_flag, 
            dm.extra_prop_hdf5, 
            dm.itr_hdf5_path, 
            dm.itr_extra_prop_path, 
            current_iter
        )
    
    print(f"\n Completed backward pass for iteration {current_iter} \n")


# Hydra-decorated version for configuration file loading
@hydra_main(version_base=None, config_path="../../configs", config_name="experiment_config")
def backward_main_hydra(cfg: DictConfig):
    """
    Hydra entry point for running backward pass with configuration files.
    """
    backward_main(cfg=cfg)


if __name__ == "__main__":
    backward_main_hydra()
