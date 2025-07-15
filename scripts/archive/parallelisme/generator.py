import os
import time
import socket

def generate(input_folder, output_folder):

    start_time = time.time()
    hostname = socket.gethostname()
    print(f"Starting generation on {hostname} from {input_folder} to {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".xyz"):
            with open(os.path.join(input_folder, filename), "r") as infile:
                lines = infile.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 4:
                    x, y, z = map(float, parts[1:])
                    new_lines.append(f"{parts[0]} {-x:.3f} {-y:.3f} {-z:.3f}\n")

            with open(os.path.join(output_folder, filename), "w") as outfile:
                outfile.writelines(new_lines)

    end_time = time.time()
    print(f"Finished generation on {hostname} in {end_time - start_time:.2f} seconds")
