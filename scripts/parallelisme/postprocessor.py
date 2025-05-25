import os
import sys
import time
import socket

def process_file(file_path, output_folder):
    start_time = time.time()
    hostname = socket.gethostname()
    filename = os.path.basename(file_path)
    print(f"Starting processing {filename} on {hostname}")

    with open(file_path, "r") as infile:
        lines = infile.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            x, y, z = map(float, parts[1:])
            new_lines.append(f"{parts[0]} {x*2:.3f} {y*2:.3f} {z*2:.3f}\n")

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, filename), "w") as outfile:
        outfile.writelines(new_lines)

    end_time = time.time()
    print(f"Finished processing {filename} on {hostname} in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    file_path = sys.argv[1]
    output_folder = sys.argv[2]
    process_file(file_path, output_folder)
