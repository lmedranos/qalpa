import os
import sys 
import shutil

if __name__ == "__main__":
    """
    Filter molecules based on SAScore from a file.
    This script reads a file containing molecule names and their SAScore values,
    and copies the molecules with a SAScore above a specified threshold to an output directory.
    """
    
    score_threshold = 5     # need to be moved to configration file
    ipath = sys.argv[1]
    opath = sys.argv[2] 
  
    filteration_file = os.path.join(ipath,'sascore.txt')
    if not os.path.exists(filteration_file):
        print(f"Filter file {filteration_file} does not exist.")
        sys.exit(1)

    with open(filteration_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            print(f"Invalid line format: {line.strip()}")
            continue
        molecule_name, sascore = parts
        try:
            sascore = float(sascore)
            if sascore >= score_threshold:
                # copy this molecule to the output path
                src_file = os.path.join(ipath, molecule_name)
                dst_file = os.path.join(opath, molecule_name)
                shutil.copy(src_file, dst_file)
        except ValueError:
            print(f"Invalid SAScore value: {sascore} for molecule {molecule_name}")