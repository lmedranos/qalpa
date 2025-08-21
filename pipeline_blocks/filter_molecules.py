import os
import sys 
import shutil
import argparse
from pathlib import Path
import json


# This function will be deprecated in favor of the MolecularFilter class
def run_filter_molecules(input, output):
    """    Filter molecules based on SAScore from a file.
    This function reads a file containing molecule names and their SAScore values,
    and copies the molecules with a SAScore above a specified threshold to an output directory.
    """
    # Define the accepted tiers for filtering
    # accepted_tires = {"tier_3", "tier_2"}     # need to be moved to configration file
    # accepted_tires = {"tier_4", "tier_3"}
    ipath = Path(input)
    opath = Path(output)
    accepted_tires={"tier_4", "tier_3"}
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


class MolecularFilter:
    """
    Class to filter molecules based on their features.
    This class provides methods to filter molecules based on SAScore, accuracy, and QED.
    """
    def __init__(self):
        pass   

    def sacore_filter(self, input_path, output_path, sascore_file, trajectory, iter):
        """    Filter molecules based on SAScore from a file.
    This function reads a file containing molecule names and their SAScore values,
    and copies the molecules with a SAScore above a specified threshold to an output directory.
    """ 
        prop = "SAScore"
        ref_value = trajectory[prop][iter] 
        ipath = Path(input_path)
        opath = Path(output_path)

        if sascore_file is None:
            filteration_file = ipath / 'sascore.txt'
        else:
            filteration_file = Path(sascore_file)

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
                # if tier in accepted_tires:
                if float(sascore) <= ref_value:
                    # copy this molecule to the output path
                    src_file = ipath / molecule_name
                    dst_file = opath / molecule_name
                    shutil.copy(src_file, dst_file)
            except ValueError:
                print(f"Invalid SAScore value: {sascore} for molecule {molecule_name}")

    def accuracy_filter(self, input_path, output_path, prop, pjsons, trajectory, iter, threshold=0.1):
        """ Filter molecules based on accuracy from a file.
        This function filters molecules based on a property value (e.g., eMBD) from a trajectory.
        It checks if the property value of each molecule is within a specified threshold
        of the reference value from the trajectory at a given iteration.
        The molecules that meet the criteria are copied to the output directory.
        """

        ref_value = trajectory[prop][iter] 
        lower_bound = ref_value - (threshold * ref_value)
        upper_bound = ref_value + (threshold * ref_value)
        ipath = Path(input_path)
        opath = Path(output_path)
        jfiles = [f for f in pjsons.iterdir() if f.is_file() and f.suffix == '.json']  
        for jf in jfiles:
            with jf.open("r") as json_file:
                data = json.load(json_file)
            mol_name = jf.stem
            if prop in data and lower_bound <= data[prop] <= upper_bound:
                # copy this molecule to the output path
                src_file = ipath / f"{mol_name}.xyz"
                dst_file = opath / f"{mol_name}.xyz"
                shutil.copy(src_file, dst_file)

    def qed_filter(self, input_path, output_path, qed_file, trajectory, iter):
        """ Filter molecules based on QED from a file.
        This function reads a file containing molecule names and their QED values,
        and copies the molecules with a QED above a specified threshold to an output directory.
        """
        prop = "QED"
        ref_value = trajectory[prop][iter] 
        ipath = Path(input_path)
        opath = Path(output_path)

        if qed_file is None:
            filteration_file = ipath / 'qed.txt'
        else:
            filteration_file = Path(qed_file)

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
            molecule_name, qed, tier = parts
            try:
                if float(qed) >= ref_value:
                    # copy this molecule to the output path
                    src_file = ipath / molecule_name
                    dst_file = opath / molecule_name
                    shutil.copy(src_file, dst_file)
            except ValueError:
                print(f"Invalid QED value: {qed} for molecule {molecule_name}")


