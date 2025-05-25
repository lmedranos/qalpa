import os
import subprocess
import time
import socket

max_parallel_tasks = int(os.environ.get("SLURM_NTASKS", 8))
NUM_ITERATIONS = 3

start_loop = time.time()
hostname = socket.gethostname()
print(f"Starting iterations  on {hostname}")

for n in range(NUM_ITERATIONS):
    print(f"\n Iteration {n}")
    cur_folder = f"data/curated/cur_{n}"
    eval_folder = f"data/evaluated/eval_{n+1}"
    next_cur_folder = f"data/curated/cur_{n+1}"

    # Step 1: Run generator
    print(f" Running generator for {cur_folder} -> {eval_folder}")
    import generator
    generator.generate(cur_folder, eval_folder)

  

    # Step 2: Run post-processor (parallel via srun)
    print(f"  Running post-processor for {eval_folder} -> {next_cur_folder}")
    xyz_files = [f for f in os.listdir(eval_folder) if f.endswith(".xyz")]

    processes = []
    for f in xyz_files:
        cmd = [
            "srun", "--ntasks=1", "--cpus-per-task=1", "--exclusive",
            "python3", "postprocessor.py",
            os.path.join(eval_folder, f),
            next_cur_folder
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

    print(f" Iteration {n} done")

end_loop = time.time()
print(f"Finished workflow on {hostname} Total main loop time: {end_loop - start_loop:.2f} seconds")
