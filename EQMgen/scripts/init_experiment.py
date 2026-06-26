"""
EQMgen.scripts.init_experiment - Initialize a new experiment with configuration scaffolding

This module provides tools to quickly create new experiments with proper directory structure
and configuration templates.

Usage:
    From Python:
        from EQMgen.scripts import init_experiment
        init_experiment("my_experiment")
    
    From CLI:
        python init_experiment.py --name my_experiment
        python init_experiment.py my_experiment
"""

import argparse
import shutil
from pathlib import Path


def init_experiment(exp_name: str, base_path: Path = None, verbose: bool = True) -> Path:
    """
    Initialize a new experiment with directory structure and configuration template.
    
    Creates:
    - Experiments/{exp_name}/configs/experiment_config.yaml
    - Experiments/{exp_name}/data/
    - Experiments/{exp_name}/results/
    - Experiments/{exp_name}/logs/
    
    Args:
        exp_name: Name of the experiment
        base_path: Base path to project root (default: auto-detected from script location)
        verbose: Print status messages (default: True)
    
    Returns:
        Path to the created experiment directory
    
    Raises:
        FileExistsError: If experiment already exists
        FileNotFoundError: If template config not found
    """
    
    # Auto-detect project root if not provided
    if base_path is None:
        # Get project root (3 levels up from this script)
        script_dir = Path(__file__).parent
        base_path = script_dir.parent.parent
    
    base_path = Path(base_path)
    
    # Define paths
    experiments_dir = base_path / "Experiments"
    data_dir = base_path / "Data"
    
    exp_dir = experiments_dir / exp_name
    exp_configs_dir = exp_dir / "configs"
    template_config = experiments_dir / "template" / "configs" / "experiment_config.yaml"
    exp_jobs_dir = exp_dir / "jobs"
    template_forward_job = experiments_dir / "template" / "jobs" / "forward_job.sh"
    template_backward_job = experiments_dir / "template" / "jobs" / "backward_job.sh"
    
    exp_data_dir = data_dir / exp_name
    
    # Check if experiment already exists
    if exp_dir.exists():
        raise FileExistsError(f"Experiment '{exp_name}' already exists at {exp_dir}")
    
    # Check if template exists
    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found at {template_config}")
    if not template_forward_job.exists():
        raise FileNotFoundError(f"Template forward job script not found at {template_forward_job}")
    if not template_backward_job.exists():
        raise FileNotFoundError(f"Template backward job script not found at {template_backward_job}")
    
    # Create directories
    if verbose:
        print(f"Creating experiment: {exp_name}")
    
    exp_dir.mkdir(parents=True, exist_ok=False)
    exp_configs_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "data").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "jobs").mkdir(parents=True, exist_ok=True)
    exp_data_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  ✓ Created directory structure at {exp_dir}")
    
    # Copy template config
    config_dest = exp_configs_dir / "experiment_config.yaml"
    shutil.copy2(template_config, config_dest)
    # Copy template job scripts
    shutil.copy2(template_forward_job, exp_jobs_dir / "forward_job.sh")
    shutil.copy2(template_backward_job, exp_jobs_dir / "backward_job.sh")
    
    if verbose:
        print(f"  ✓ Copied config template to {config_dest}")
        print(f"  ✓ Copied job script templates to {exp_jobs_dir}")
    
    # Read and customize the config
    config_content = config_dest.read_text()
    
    # Replace experiment name in the config
    config_content = config_content.replace("allostery", exp_name)
    config_content = config_content.replace(
        f'exp_data_path: ${{paths.experiments}}/allostery/data',
        f'exp_data_path: ${{paths.experiments}}/{exp_name}/data'
    )
    
    config_dest.write_text(config_content)
    
    if verbose:
        print(f"  ✓ Updated config with experiment name: {exp_name}")
    
    if verbose:
        print(f"\n✓ Experiment '{exp_name}' initialized successfully!")
        print(f"\nNext steps:")
        print(f"  1. Edit config: {config_dest}")
        print(f"     - Set base_dataset_path to your dataset")
        print(f"     - Customize data.props and data.scores if needed")
        print(f"     - Adjust training/geometry/properties as needed")
        print(f"\n  2. Run experiment:")
        print(f"     python EQMgen/scripts/run_experiment.py \\")
        print(f"         --config-path Experiments/{exp_name}/configs \\")
        print(f"         --config-name experiment_config")
    
    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize a new experiment with configuration scaffolding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python init_experiment.py my_experiment
  python init_experiment.py --name my_experiment --base-path /path/to/project
        """
    )
    
    parser.add_argument(
        "name",
        nargs="?",
        help="Name of the experiment to create"
    )
    parser.add_argument(
        "--name",
        dest="exp_name",
        help="Name of the experiment to create (alternative to positional argument)"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        help="Base path to project root (auto-detected if not provided)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    # Get experiment name from positional or named argument
    exp_name = args.name or args.exp_name
    
    if not exp_name:
        parser.print_help()
        print("\nError: experiment name is required")
        exit(1)
    
    try:
        init_experiment(
            exp_name=exp_name,
            base_path=args.base_path,
            verbose=not args.quiet
        )
    except (FileExistsError, FileNotFoundError) as e:
        print(f"Error: {e}")
        exit(1)
