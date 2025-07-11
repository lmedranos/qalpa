import subprocess
import os
import time


def parallelize(script, input_dir, aux_dir, max_parallel_tasks=1):
    """
    Parallelize a function over an iterable using multiprocessing.
    This function runs a specified script in parallel for each file in the input directory.
    It uses `srun` to manage parallel tasks, allowing for a specified maximum number of tasks to run concurrently.
    It exports necessary environment variables to the subprocesses and waits for all tasks to complete before returning.

    Parameters:
    ---------------
        - script: The function to parallelize.
        - input_dir: Directory containing input files.
        - aux_dir: auxiliary Directory to store output files or to load config.
        - max_parallel_tasks: Maximum number of parallel tasks to run at once.
    """

    export_vars = "ALL,ROOT={},PIPELINE={},DATA={},MODEL_REGISTRY={},MODEL_CODE={},CONFIG={}".format(
    os.environ["ROOT"],
    os.environ["PIPELINE"],
    os.environ["DATA"],
    os.environ["MODEL_REGISTRY"],
    os.environ["MODEL_CODE"],
    os.environ["CONFIG"]
    )

    if script.endswith(".py"):
        exec_cmd = ["python3", script]
        exec_flag = False
    else:
        exec_cmd = script
        exec_flag = True

    # Find all files in the input directory
    files = [f for f in os.listdir(input_dir)]
    # Start parallel processing
    processes = []
    print(f"max_parallel_tasks: {max_parallel_tasks}")
    for f in files:
        base_cmd = [
            "srun", "--nodes=1", "--ntasks=1", 
            "--cpus-per-task=6", "--cpu-bind=cores", #"--mem-bind=local"   
            f"--export={export_vars}",
        ]
        cmd = " ".join(base_cmd + [exec_cmd]) + " &" if exec_flag else base_cmd + exec_cmd + [os.path.join(input_dir, f), aux_dir]
        cwd_path = os.path.join(input_dir, f) if exec_flag else None
        p = subprocess.Popen(cmd, shell=exec_flag, cwd= cwd_path)
        processes.append(p)
        print(f"number of processes: {len(processes)}")
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

    