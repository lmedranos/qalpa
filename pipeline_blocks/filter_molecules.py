import os
import sys 
import shutil
import argparse
from pathlib import Path
import json
from functools import reduce



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
    def __init__(self, trajectory):
        self.trajectory = trajectory
        pass   

    
    def score_filter(self, input_path, output_path, score_name, score_file, iter):
        """    Filter molecules based on a score from a file.
    This function reads a file containing molecule names and their score values,
    and copies the molecules with a score above or below a specified threshold to an output directory.
    """
        ref_value = self.trajectory.scores[score_name][iter] 
        ipath = Path(input_path)
        opath = Path(output_path)

      
        if not score_file.exists():
            print(f"Filter file {score_file} does not exist.")
            sys.exit(1)

        with score_file.open("r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                print(f"Invalid line format: {line.strip()}")
                continue
            molecule_name = parts[0]
            score = parts[1]
            try:
                if score != 'None':
                    src_file = ipath / molecule_name
                    dst_file = opath / molecule_name
                    if score_name.lower() == 'sascore' and float(score) <= ref_value:
                        shutil.copy(src_file, dst_file)
                    elif score_name.lower() == 'qed' and float(score) >= ref_value:
                        shutil.copy(src_file, dst_file)
            except ValueError:
                print(f"Invalid score value: {score} for molecule {molecule_name}")


    def accuracy_filter(self, input_path, output_path, prop, pjsons, iter, threshold=0.1):
        """ Filter molecules based on accuracy from a file.
        This function filters molecules based on a property value (e.g., eMBD) from a trajectory.
        It checks if the property value of each molecule is within a specified threshold
        of the reference value from the trajectory at a given iteration.
        The molecules that meet the criteria are copied to the output directory.
        """
        prop_dict = {
            "eMBD": ("SUM", "MBD"),  # nested path
            "HLgap": ("HOMO-LUMO gap",),  # top-level
            "DIP": ("Absolute Total Dipole (eAng):",)  # top-level
        }
        
        get_value = lambda data, keys: reduce(
            lambda d, k: d.get(k) if isinstance(d, dict) else None,
            keys if isinstance(keys, (list, tuple)) else (keys,),
            data)
        
        ref_value = float(self.trajectory.properties[prop][iter])
        delta = threshold * abs(ref_value)
        lower_bound = ref_value - delta
        upper_bound = ref_value + delta
        lower_bound, upper_bound = sorted((lower_bound, upper_bound )) 
        ipath = Path(input_path)
        opath = Path(output_path)
        jfiles = [f for f in pjsons.iterdir() if f.is_file() and f.suffix == '.json']  
        for jf in jfiles:
            with jf.open("r") as json_file:
                data = json.load(json_file)
            mol_name = jf.stem
            keys = prop_dict[prop]
            value = get_value(data, keys)
            if value is not None and lower_bound <= value <= upper_bound:
                # copy this molecule to the output path
                src_file = ipath / f"{mol_name}.xyz"
                dst_file = opath / f"{mol_name}.xyz"
                shutil.copy(src_file, dst_file)

 