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
    env = os.environ.copy()
    #env["LD_LIBRARY_PATH"] = "/software/rome/r24.04/imkl/2023.2.0/mkl/2023.2.0/lib/intel64:" + env.get("LD_LIBRARY_PATH", "")


    base_cmd = [
        "srun", "--ntasks=1", # "--mpi=pmi2", "--cpus-per-task=6",
        "--exclusive",# "--cpu-bind=cores",
        "--export=ALL", 
        f"--export=CONFIG={os.environ['CONFIG']}",
    ]

    if str(script_path).endswith(".py"):
        python_exec = Path(os.environ["CONDA_PREFIX"]) / "bin" / "python"
        exec_cmd = [str(python_exec), str(script_path)]
        exec_flag = False
    else:
        exec_cmd =  str(script_path)
        exec_flag = True
    # Find all files in the input directory
    paths = [f for f in input_path.iterdir()]    
    processes = []

    for fpath in paths:
        cmd = " ".join(base_cmd + [exec_cmd]) if exec_flag else base_cmd + exec_cmd + ["--input", str(fpath), "--output", str(output_path)]
        cwd_path = str(fpath) if exec_flag else None
        p = subprocess.Popen(cmd, shell=exec_flag, cwd= cwd_path, env=env)
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

    