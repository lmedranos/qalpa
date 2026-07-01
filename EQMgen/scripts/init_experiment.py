"""
EQMgen.scripts.init_experiment - Initialize a new experiment with configuration 

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


def init_experiment(exp_name: str, dataset_path: str , base_path: str = None, verbose: bool = True):
    """
    Initialize a new experiment with directory structure and configuration template.
    
    Creates:
    - Experiments/{exp_name}/configs/experiment_config.yaml
    - Experiments/{exp_name}/jobs/
    - Experiments/{exp_name}/results/
    - Experiments/{exp_name}/logs/
    - Data/{exp_name}/ (for experiment-specific data)
    
    Args:
        exp_name: Name of the experiment
        dataset_path: Path to the dataset to be used for the experiment
        base_path: Base path to project root (default: auto-detected from script location)
        verbose: Print status messages (default: True)
    
    Raises:
        FileExistsError: If experiment already exists
        FileNotFoundError: If template config not found
        FileNotFoundError: If template job scripts not found
        FileNotFoundError: If dataset path does not exist
    """
    
    # Auto-detect project root if not provided
    if base_path is None:
        # Get project root (2 levels up from this script)
        script_dir = Path(__file__).parent
        base_path = script_dir.parent.parent
    
    base_path = Path(base_path)
    
    # Define paths
    experiments_dir = base_path / "Experiments"
    exp_data_dir = base_path / "Data" / exp_name
    
    # Check if experiment already exists
    v = 0
    while (experiments_dir / f"{exp_name}_{v}").exists():
        v += 1
    exp_name = f"{exp_name}_{v}"                # Append version number to experiment name
    exp_dir = experiments_dir / exp_name
    exp_jobs_dir = exp_dir / "jobs"
    exp_configs_dir = exp_dir / "configs"

    template_config = experiments_dir / "templates" / "configs" / "experiment_config.yaml"
    template_forward_job = experiments_dir / "templates" / "jobs" / "eqmgen_forward.sh"
    template_backward_job = experiments_dir / "templates" / "jobs" / "eqmgen_backward.sh"
    # Check if template exists
    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found at {template_config}")
    if not template_forward_job.exists():
        raise FileNotFoundError(f"Template forward job script not found at {template_forward_job}")
    if not template_backward_job.exists():
        raise FileNotFoundError(f"Template backward job script not found at {template_backward_job}")
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
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
    shutil.copy2(template_forward_job, exp_jobs_dir / "eqmgen_forward.sh")
    shutil.copy2(template_backward_job, exp_jobs_dir / "eqmgen_backward.sh")
    
    if verbose:
        print(f"  ✓ Copied config template to {config_dest}")
        print(f"  ✓ Copied job script templates to {exp_jobs_dir}")
    
    # Read and customize the config
    config_content = config_dest.read_text()
    
    # Absolute paths injected into the config and job scripts
    repo_root_abs = str(base_path.resolve())
    exp_config_dir_abs = str(exp_configs_dir.resolve())

    # Replace experiment name in the config
    config_content = config_content.replace("template_experiment", exp_name)
    config_content = config_content.replace("template_dataset", str(dataset_path))
    # Inject the repo root so Hydra can resolve the absolute defaults (searchpath)
    config_content = config_content.replace("@@REPO_ROOT@@", repo_root_abs)
    config_dest.write_text(config_content)

    # Customize the copied job scripts with absolute repo root and experiment config dir
    for job_script in (exp_jobs_dir / "eqmgen_forward.sh", exp_jobs_dir / "eqmgen_backward.sh"):
        job_content = job_script.read_text()
        job_content = job_content.replace("@@REPO_ROOT@@", repo_root_abs)
        job_content = job_content.replace("@@EXP_CONFIG_DIR@@", exp_config_dir_abs)
        job_script.write_text(job_content)
    
    if verbose:
        print(f"  ✓ Updated config with experiment name: {exp_name}")
        print(f"  ✓ Wired job scripts to config dir: {exp_config_dir_abs}")
    
    if verbose:
        print(f"\n✓ Experiment '{exp_name}' initialized successfully!")
        print(f"\nNext steps:")
        print(f"  1. Edit config: {config_dest}")
        print(f"     - Set base_dataset_path to your dataset")
        print(f"     - Customize data.props and data.scores if needed")
        print(f"     - Adjust training/geometry/properties as needed")
        print(f"\n  2. Run experiment: Alternatively")
        print(f"     - submit  {exp_jobs_dir / 'eqmgen_forward.sh'}")
        print(f"     - submit  {exp_jobs_dir / 'eqmgen_backward.sh'}")
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize a new experiment with configuration scaffolding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python init_experiment.py my_experiment
  python init_experiment.py --name my_experiment --dataset-path /path/to/dataset
  python init_experiment.py --name my_experiment --dataset-path /path/to/dataset --base-path /path/to/root
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
        "--dataset-path",
        type=Path,
        help="Path to the dataset to be used for the experiment"
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
            dataset_path=args.dataset_path,
            base_path=args.base_path,
            verbose=not args.quiet
        )
    except (FileExistsError, FileNotFoundError) as e:
        print(f"Error: {e}")
        exit(1)
