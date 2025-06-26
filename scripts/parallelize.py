import subprocess
import os
import time


def parallelize(script, input_dir, output_dir, max_parallel_tasks=1):
    """
    Parallelize a function over an iterable using multiprocessing.
    This function runs a specified script in parallel for each file in the input directory.
    It uses `srun` to manage parallel tasks, allowing for a specified maximum number of tasks to run concurrently.
    It exports necessary environment variables to the subprocesses and waits for all tasks to complete before returning.

    Parameters:
    ---------------
        - script: The function to parallelize.
        - input_dir: Directory containing input files.
        - output_dir: Directory to store output files.
        - max_parallel_tasks: Maximum number of parallel tasks to run at once.
    """
    # 

    export_vars = "ALL,ROOT={},PIPELINE={},DATA={},MODEL_REGISTRY={},MODEL_CODE={},CONFIG={}".format(
    os.environ["ROOT"],
    os.environ["PIPELINE"],
    os.environ["DATA"],
    os.environ["MODEL_REGISTRY"],
    os.environ["MODEL_CODE"],
    os.environ["CONFIG"],
)
    # Script to run
    script_dir = os.environ["PIPELINE"]

    if not script.endswith(".py"):
        script += ".py"
    script_path = os.path.join(script_dir, script)

    # Find all .xyz files in the input directory
    xyz_files = [f            for f in os.listdir(input_dir) if f.endswith(".xyz")]

    # Start parallel processing
    processes = []
    for f in xyz_files:
        cmd = [
            "srun", "--ntasks=1", "--cpus-per-task=1", "--exclusive",   
            f"--export={export_vars}",
            "python3", script_path,
            os.path.join(input_dir, f),
            output_dir,
        ]
        p = subprocess.Popen(cmd)
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

    