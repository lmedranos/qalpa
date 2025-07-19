import os
import sys 
import shutil
import argparse
from pathlib import Path



def run_filter_molecules(input, output):
    """    Filter molecules based on SAScore from a file.
    This function reads a file containing molecule names and their SAScore values,
    and copies the molecules with a SAScore above a specified threshold to an output directory.
    """
    # Define the accepted tiers for filtering
    accepted_tires = {"tier_3", "tier_2"}     # need to be moved to configration file

    ipath = Path(input)
    opath = Path(output)
  
    filteration_file = ipath / 'sascore.txt'
    if not filteration_file.exists():
        print(f"Filter file {filteration_file} does not exist.")
        sys.exit(1)

    with filteration_file.open("r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            print(f"Invalid line format: {line.strip()}")
            continue
        molecule_name, sascore, tier = parts
        try:
            if tier in accepted_tires:
                # copy this molecule to the output path
                src_file = ipath / molecule_name
                dst_file = opath / molecule_name
                shutil.copy(src_file, dst_file)
        except ValueError:
            print(f"Invalid SAScore value: {sascore} for molecule {molecule_name}")


if __name__ == "__main__":
    """
    Filter molecules based on SAScore from a file.
    This script reads a file containing molecule names and their SAScore values,
    and copies the molecules with a SAScore above a specified threshold to an output directory.
    """
    parser = argparse.ArgumentParser(description="Filter molecules based on SAScore.")
    parser.add_argument("--input", type=str, help=" Path to the input directory containing molecules and SAScore file")
    parser.add_argument("--output", type=str, help=" Output directory for filtered molecules")
    args = parser.parse_args()
    run_filter_molecules(args.input, args.output)