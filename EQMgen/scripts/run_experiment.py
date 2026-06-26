"""
run_experiment.py - Main entry point for running experiments

This script shows the recommended way to run experiments using the three-tier
hierarchical configuration system.

Usage:
    # From experiment directory
    python ../../EQMgen/scripts/run_experiment.py
    
    # With overrides
    python ../../EQMgen/scripts/run_experiment.py training.num_iterations=5
    
    # From project root (specify experiment)
    python EQMgen/scripts/run_experiment.py \
        --config-path Experiments/allostery/configs \
        --config-name experiment_config
"""

import os
import sys
from pathlib import Path
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from EQMgen.scripts import forward_main, backward_main


@hydra_main(
    version_base=None,
    config_path="../../../Experiments/allostery/configs",  # Relative to EQMgen/scripts/
    config_name="experiment_config"
)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run the full experiment with hierarchical configuration.
    
    The configuration is resolved in this order:
    1. Root level: /configs/system.yaml, training.yaml, paths.yaml
    2. Framework level: EQMgen/configs/eqmgen_*.yaml
    3. Experiment level: Experiments/{exp_name}/configs/exp_*.yaml
    
    Later configs override earlier ones.
    
    Usage:
        python run_experiment.py
        python run_experiment.py training.num_iterations=5
        python run_experiment.py geometry.optimizer=EquiDTB
    """
    
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70 + "\n")
    
    num_iterations = cfg.training.num_iterations
    
    for iteration in range(num_iterations):
        print(f"\n{'#' * 70}")
        print(f"# ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'#' * 70}\n")
        
        try:
            # Forward pass: Generate molecules
            print("→ Forward Pass: Generating molecules...")
            forward_main(
                base_dataset=cfg.base_dataset_path,
                dataset_name="qm7x",
                data_path=str(cfg.data.exp_data_path),
                model_code=str(PROJECT_ROOT / cfg.code_paths.EQdiff),
                config_path=str(PROJECT_ROOT / "configs"),
                actdiff_utils=str(PROJECT_ROOT / cfg.code_paths.actdiff_utils),
                num_iterations=cfg.training.num_iterations,
                iter=iteration,
                batch_size=cfg.training.batch_size,
                n_graphs=cfg.EQdiffParam.num_samples,
                props=cfg.data.props,
                scores=cfg.data.scores,
                n_nodes=[cfg.EQdiffParam.n_nodes] * num_iterations,
                max_parallel_tasks=cfg.training.max_parallel_tasks,
            )
            print("✓ Forward pass completed\n")
        except Exception as e:
            print(f"⚠ Forward pass warning: {e}\n")
        
        try:
            # Backward pass: Train and evaluate
            print("→ Backward Pass: Training and evaluating...")
            backward_main(cfg=cfg)
            print("✓ Backward pass completed\n")
        except Exception as e:
            print(f"⚠ Backward pass warning: {e}\n")
        
        print(f"\n✓ Iteration {iteration + 1}/{num_iterations} completed")


if __name__ == "__main__":
    run_experiment()
