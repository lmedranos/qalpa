import subprocess
import os
import time
from pathlib import Path


def parallelize(script_path, input_path, output_path, max_parallel_tasks=1):
    """
    Parallelize a function over an iterable using multiprocessing.
    This function runs a specified script in parallel for each file in the input directory.
    It uses `srun` to manage parallel tasks, allowing for a specified maximum number of tasks to run concurrently.
    It exports necessary environment variables to the subprocesses and waits for all tasks to complete before returning.

    Parameters:
    ---------------
        - script: Path to the function to parallelize.
        - input_dir: Path of the directory containing input files.
        - output_path: Path to store output files.
        - max_parallel_tasks: Maximum number of parallel tasks to run at once.
    """
    
    # Setup environment for parallel calculations of molecules
    # env = os.environ.copy()
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) #*int(os.environ.get("SLURM_TASKS_PER_NODE"))*int(os.environ.get("SLURM_NNODES"))// (max_parallel_tasks)  
    # env["OMP_NUM_THREADS"] = str(cpus_per_task)
    # env["MKL_NUM_THREADS"] = str(cpus_per_task)
    # env["OPENBLAS_NUM_THREADS"] = str(cpus_per_task)
    # env["NUMEXPR_NUM_THREADS"] = str(cpus_per_task)
    # os.environ.update(env)
    
    
    shell_flag = not str(script_path).endswith(".py")

    base_cmd = [
        "srun", "--ntasks=1", f"--cpus-per-task={cpus_per_task}",
        "--exclusive",
        f"--export=ALL,CONFIG={os.environ['CONFIG']}",
    ]

    paths = [f for f in input_path.iterdir()]    
    processes = []

    for fpath in paths:
        if shell_flag:
            cmd = " ".join(base_cmd + [str(script_path), ">", str(fpath / output_path)])
        else:
            cmd = base_cmd + ["python3", str(script_path), "--input", str(fpath), "--output", str(output_path)]
        
        p = subprocess.Popen(cmd, shell=shell_flag, cwd = str(fpath) if shell_flag else None)
        processes.append(p)
        # Control max parallelism
        if len(processes) >= max_parallel_tasks:
            # Wait for any to finish before launching more
            finished_proc = None
            while finished_proc is None:
                for p in processes:
                    ret = p.poll()
                    if ret is not None:  # process finished
                        finished_proc = p
                        break
                time.sleep(0.1)
            processes.remove(finished_proc)
    # Wait for all post-processor jobs to finish
    for p in processes:
        p.wait()

    