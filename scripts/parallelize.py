import subprocess
import os
import time
from pathlib import Path


def parallelize(script_path, input_path, aux_path, max_parallel_tasks=1):
    """
    Parallelize a function over an iterable using multiprocessing.
    This function runs a specified script in parallel for each file in the input directory.
    It uses `srun` to manage parallel tasks, allowing for a specified maximum number of tasks to run concurrently.
    It exports necessary environment variables to the subprocesses and waits for all tasks to complete before returning.

    Parameters:
    ---------------
        - script: Path to the function to parallelize.
        - input_dir: Path of the directory containing input files.
        - aux_dir: Auxiliary Path to store output files or to load config.
        - max_parallel_tasks: Maximum number of parallel tasks to run at once.
    """
    
    base_cmd = [
        "srun", "--ntasks=1", # "--mpi=pmi2",
        "--cpus-per-task=3", "--exclusive",# "--cpu-bind=cores", 
        f"--export={os.environ["CONFIG"]}",
    ]

    if script_path.suffix == ".py":
        exec_cmd = ["python3", str(script_path)]
        exec_flag = False
    else:
        exec_cmd =  str(script_path)
        exec_flag = True

    # Find all files in the input directory
    paths = [f for f in input_path.iterdir() if f.is_file()]    
    processes = []

    for fpath in paths:
        cmd = " ".join(base_cmd + [exec_cmd])  if exec_flag else base_cmd + exec_cmd + [str(fpath), str(aux_path)]
        cwd_path = str(fpath) if exec_flag else None
        p = subprocess.Popen(cmd, shell=exec_flag, cwd= cwd_path)
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

    