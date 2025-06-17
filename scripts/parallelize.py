import subprocess
import os
import time

def parallelize(script, input_dir, output_dir, n_jobs=1):
    """
    Parallelize a function over an iterable using multiprocessing.

    Parameters:
    - script: The function to parallelize.
    - input_dir: Directory containing input files.
    - output_dir: Directory to store output files.
    - n_jobs: Number of parallel jobs (default is 1, meaning no parallelization).

    Returns:
    A list of results from applying the function to each item in the iterable.
"""
    # 
    export_vars = "ALL,ROOT={},DATA={},MODEL={},CODE={},CONFIG={}".format(
    os.environ["ROOT"],
    os.environ["DATA"],
    os.environ["MODEL"],
    os.environ["CODE"],
    os.environ["CONFIG"],
)
    max_parallel_tasks = int(os.environ.get("SLURM_NTASKS", n_jobs))
    # Script to run
    code_dir = os.environ["CODE"]
    if not script.endswith(".py"):
        script += ".py"
    script_path = os.path.join(code_dir, script)

    # Find all .xyz files in the input directory
    xyz_files = [f for f in os.listdir(input_dir) if f.endswith(".xyz")]

    # Start parallel processing
    processes = []
    for f in iterable:
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

    