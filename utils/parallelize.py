import subprocess
import signal
import os
import time
from pathlib import Path
from collections import deque
import sys
from mpi4py import MPI



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

# This function is going to be deprecated in favor of node_parallelize function 
def parallelize(script_path, input_path, output_path, max_parallel_tasks=None, ntasks=None, nthreads=None):
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
    n_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    max_parallel_tasks = max_parallel_tasks if max_parallel_tasks else n_nodes   
    tasks_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE", 1).split('(')[0])
    ntasks = ntasks if ntasks and (ntasks <= tasks_per_node) else tasks_per_node
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) 
    print(f"Parallelizing with {ntasks} tasks and {cpus_per_task} cpus per task (max {max_parallel_tasks} parallel tasks)")
    
    # Setup environment for parallel calculations of molecules
    # env = os.environ.copy()
    # env["OMP_NUM_THREADS"] = str(cpus_per_task)
    # env["MKL_NUM_THREADS"] = str(cpus_per_task)
    # env["OPENBLAS_NUM_THREADS"] = str(cpus_per_task)
    # env["NUMEXPR_NUM_THREADS"] = str(cpus_per_task)
    # os.environ.update(env)
    
    shell_flag = not str(script_path).endswith(".py")

    base_cmd = [
        "srun", "--nodes=1", f"--ntasks={ntasks}", f"--cpus-per-task={cpus_per_task}",
        "--exclusive",
        f"--export=ALL,CONFIG={os.environ['CONFIG']}",
    ]

    paths = [f for f in input_path.iterdir()]    
    processes = []

    for fpath in paths:
        if shell_flag:
            if skip_processing(fpath / output_path):
                print(f"Skipping {fpath}")
                continue
            else:
                cmd = " ".join(base_cmd + [str(script_path), ">", str(fpath / output_path)])
        else:
            cmd = base_cmd + ["python3", str(script_path), "--input", str(fpath), "--output", str(output_path)]
        
        p = subprocess.Popen(cmd, shell=shell_flag, cwd = str(fpath) if shell_flag else None)
        processes.append(p)
        # Control max parallelism
        if len(processes) >= int(max_parallel_tasks):
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
        
    
def rank_parallelize(script_path, input_path, output_path):
    """
    Parallelize a function over an iterable using MPI.
    This function runs a specified script in parallel for each file in the input directory.

    Parameters:
    ---------------
        - script: Path to the function to parallelize.
        - input_dir: Path of the directory containing input files.
        - output_path: Path to store output files.
    """ 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) 

    # # Setup environment for parallel calculations of molecules
    # env = os.environ.copy()
    # env["OMP_NUM_THREADS"] = str(cpus_per_task)
    # env["MKL_NUM_THREADS"] = str(cpus_per_task)
    # # env["OPENBLAS_NUM_THREADS"] = str(cpus_per_task)
    # # env["NUMEXPR_NUM_THREADS"] = str(cpus_per_task)
    # os.environ.update(env)
    
    shell_flag = not str(script_path).endswith(".py")

    files = sorted([f for f in input_path.iterdir() if f.is_file()])
    files_for_rank = files[rank::size]
    print(f"[Rank {rank}/{size}] Assigned {len(files_for_rank)} files")
    
    for fpath in files_for_rank:
        if shell_flag:
            if skip_processing(fpath / output_path, marker= " Have a nice day."):
                print(f"[Rank {rank}] Skipping {fpath}")
                continue
            else:
                cmd = " ".join([str(script_path), ">", str(fpath / output_path)])
        else:
            # if skip_processing (output_path / fpath.name, marker= None):
            #     print(f"Skipping {fpath}")
            #     continue
            # else:
            cmd = ["python3", str(script_path), "--input", str(fpath), "--output", str(output_path)]
                # print(f"[Rank {rank}] Running {cmd}")
        
        r = subprocess.run(cmd, shell=shell_flag, cwd = str(fpath) if fpath.is_dir() else None)
        
        if r.returncode != 0:
            print(f"[Rank {rank}] Error processing {fpath}", file=sys.stderr)
            
    # Wait until all ranks finish
    comm.Barrier()
    if rank == 0:
        print("All files processed on all ranks.")
            
def node_parallelize(script_path, input_path, output_path, max_parallel_tasks=None, nranks=None, timeout_hours=5):
    """
    Parallelize a function over an iterable using multiprocessing.
    It parallelizes files processing over nodes.
    It may use MPI internally to parallelize computation of each file over nranks, if module supports it.
  
    Parameters:
    ---------------
        - script: Path to the function to parallelize.
        - input_dir: Path of the directory containing input files.
        - output_path: Path to store output files.
        - max_parallel_tasks: Maximum number of parallel tasks to run at once.
        - nranks: Number of MPI ranks to use per task.
    """ 
    # Get SLURM parameters
    n_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    max_parallel_tasks = max_parallel_tasks if max_parallel_tasks else n_nodes   
    tasks_per_node = int(os.environ.get("SLURM_TASKS_PER_NODE", 1).split('(')[0])
    nranks = nranks if nranks and (nranks <= tasks_per_node) else tasks_per_node
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) 

    print(f"Parallelizing with {nranks} tasks and {cpus_per_task} cpus per task (max {max_parallel_tasks} parallel tasks)")
    
    # # Setup environment for parallel calculations of molecules
    # env = os.environ.copy()
    # env["OMP_NUM_THREADS"] = str(cpus_per_task)
    # env["MKL_NUM_THREADS"] = str(cpus_per_task)
    # # env["OPENBLAS_NUM_THREADS"] = str(cpus_per_task)
    # # env["NUMEXPR_NUM_THREADS"] = str(cpus_per_task)
    # os.environ.update(env)
    
    shell_flag = not str(script_path).endswith(".py")

    base_cmd = [
        "srun", "--nodes=1", f"--ntasks={nranks}", f"--cpus-per-task={cpus_per_task}",
        "--exclusive",
        f"--export=ALL,CONFIG={os.environ['CONFIG']}",
    ]

    paths = [f for f in input_path.iterdir()]    
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
            cmd = base_cmd + ["python3", str(script_path), "--input", str(fpath), "--output", str(output_path)]
        
        p = subprocess.Popen(cmd, shell=shell_flag, cwd = str(fpath) if fpath.is_dir() else None, preexec_fn= os.setsid if shell_flag else None)
        processes.append(p)
        start_times[p.pid] = time.time()
        proc_to_file[p.pid] = fpath.name
        # Control max parallelism
        while len(processes) >= int(max_parallel_tasks):
            time.sleep(5)
            processes, pkilled = monitor_processes(processes, start_times, timeout_hours)
            fkilled.extend([proc_to_file.get(pk.pid, "unknown") for pk in pkilled])
    # Wait for all processor jobs to finish
    while processes:
        time.sleep(5)
        processes, pkilled = monitor_processes(processes, start_times, timeout_hours)
        fkilled.extend([proc_to_file.get(pk.pid, "unknown") for pk in pkilled])
        
    # for p in processes:
    #     try:
    #         os.waitpid(p.pid, 0)  # ensures it’s truly reaped
    #     except ChildProcessError:
    #         pass
        
    return fkilled
  
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
        elif runtime_hours > timeout_hours:
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
    rank_parallelize(script_path, Path(input_path), Path(output_path))
