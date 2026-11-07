import subprocess
import signal
import os
import time
from pathlib import Path
from collections import deque
import sys
import warnings
#from mpi4py import MPI


def node_parallelize(script_path, input_path, output_path, max_parallel_tasks=None, aux_arg=None, nranks=None, timeout_hours=2, batch_size=None, batch_id=None):
    """
    Parallelize a function over an iterable using multiprocessing.
    It parallelizes files processing over nodes.
    It may use MPI internally to parallelize computation of each file over nranks, if module supports it.
  
    Parameters:
    ---------------
        - script: Path to the function to parallelize.
        - input_dir: Path of the directory containing input files.
        - output_path: Path to store output files.
        - aux_arg: Auxiliary arguments to pass to the script.
        - max_parallel_tasks: Maximum number of parallel tasks to run at once.
        - nranks: Number of MPI ranks to use per task.
    """ 
    # Get SLURM parameters
    n_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    max_parallel_tasks = max_parallel_tasks if max_parallel_tasks else n_nodes   
    tasks_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE", "1").split('(')[0])
    nranks = nranks if nranks and (nranks <= tasks_per_node) else tasks_per_node
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) 

    print(f"Parallelizing with {nranks} tasks and {cpus_per_task} cpus per task (max {max_parallel_tasks} parallel tasks)")
    
    
    shell_flag = not str(script_path).endswith(".py")

    base_cmd = [
        "srun", "--nodes=1", f"--ntasks={nranks}", f"--cpus-per-task={cpus_per_task}",
        "--exclusive", "--exact", "--distribution=block:block",
        "--mpi=pmi2",  # Intel MPI compatibility
        f"--export=ALL,DFTB_PREFIX={os.environ['DFTB_PREFIX']}",
    ]

    paths = [f for f in input_path.iterdir()]

    # Process only a patch of data if specified
    if batch_id is not None and batch_size is not None:
        start_idx = batch_id * batch_size
        end_idx = min((batch_id + 1) * batch_size, len(paths))
        paths = paths[start_idx:end_idx]
        print(f"Processing patches {batch_id}, total {len(paths)} files.")
    processes = []
    start_times = {}
    proc_to_file = {}
    fkilled = []

    for fpath in paths:
        if shell_flag:
            if skip_processing(fpath / output_path, marker= " Have a nice day."):
                print(f"Skipping {fpath}")
                continue
            else:
                cmd = " ".join(base_cmd + [str(script_path), ">", str(fpath / output_path)])
                
        else:
            # if skip_processing (output_path / fpath.name, marker= None):
            #     print(f"Skipping {fpath}")
            #     continue
            # else:
            aux_arg_cmd = ["--aux_arg"] + [str(a) for a in aux_arg] if aux_arg else []
            cmd = base_cmd + ["python3", str(script_path), "--input", str(fpath), "--output", str(output_path)] + aux_arg_cmd
            print("COMMAND:")
            print(" ".join(cmd) if isinstance(cmd, list) else cmd)
            # Recommended uptate to check
            py_base_cmd = ["--ntasks=1" if c.startswith("--ntasks") else c for c in base_cmd]
            cmd = py_base_cmd + ["python3", str(script_path), "--input", str(fpath), "--output", str(output_path)] + aux_arg_cmd
        p = subprocess.Popen(cmd, shell=shell_flag, cwd = str(fpath) if fpath.is_dir() else None, preexec_fn= os.setsid)
        # p = subprocess.Popen(cmd, shell=shell_flag, cwd = str(fpath) if fpath.is_dir() else None, preexec_fn= os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)# if shell_flag else None)
        processes.append(p)
        start_times[p.pid] = time.time()
        proc_to_file[p.pid] = fpath.name
        # Control max parallelism
        while len(processes) >= int(max_parallel_tasks):
            time.sleep(5)
            # for p in processes[:]:
            #     ret = p.poll()
            #     if ret is not None:
            #         stdout, stderr = p.communicate()
            #         print("=" * 80)
            #         print("Process finished:", p.pid)
            #         print("Return code:", ret)
            #         print("STDOUT:")
            #         print(stdout)
            #         print("STDERR:")
            #         print(stderr)
            #         print("=" * 80)
            processes, pkilled = monitor_processes(processes, start_times, timeout_hours)
            fkilled.extend([proc_to_file.get(pk.pid, "unknown") for pk in pkilled])
    # Wait for all processor jobs to finish
    
    while processes:
        time.sleep(5)
        processes, pkilled = monitor_processes(processes, start_times, timeout_hours)
        fkilled.extend([proc_to_file.get(pk.pid, "unknown") for pk in pkilled])
        
    return fkilled 


def skip_processing(ofile: Path, scope=20, marker=" Have a nice day.") -> bool:
    """Return True if the output file exists and contains the marker in last 20 lines."""
    if ofile.exists():
        if marker is None:
            return True
        else:
            if scope is None:
                # Search entire file directly without building a list
                with ofile.open('r') as f:
                    return any(marker in line for line in f)
            else:
                # Search only the last `scope` lines
                with ofile.open('r') as f:
                    last_lines = deque(f, maxlen=scope)
                return any(marker in line for line in last_lines)
    return False

    
def monitor_processes(processes, start_times, timeout_hours):
    """
    Check a list of subprocesses. Remove finished processes, kill those exceeding timeout.
    Parameters
    ----------
    processes : list
        List of subprocess.Popen objects.   
    start_times : dict
        Dictionary mapping process.pid -> start time (seconds).
    timeout_hours : float
        Maximum allowed runtime in hours.

    Returns
    -------
    processes : list
        Updated list of running processes after checking.
    """
    updated_processes = []
    killed_processes = []
    for p in processes:
        runtime_hours = (time.time() - start_times[p.pid]) / 3600.0
        # fname = proc_to_file.get(p.pid, "unknown")
        if p.poll() is not None:
            # Process finished
            continue
        elif float(runtime_hours) > float(timeout_hours):
            # Process exceeded timeout
            # print(f"Killing process {p.pid} ({fname}) after {runtime_hours:.2f}h")
            # p.kill()
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            killed_processes.append(p)
            
        else:
            # Still running
            updated_processes.append(p)

    return updated_processes, killed_processes     

if __name__ == "__main__":
    # parse args passed from main.py
    script_path, input_path, output_path = sys.argv[1:4]
    node_parallelize(script_path, Path(input_path), Path(output_path))
