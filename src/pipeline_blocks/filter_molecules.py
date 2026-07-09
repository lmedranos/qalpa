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

    
    def score_filter(self, input_path, output_path, score_name, score_file, target_indx):
        """    Filter molecules based on a score from a file.
    This function reads a file containing molecule names and their score values,
    and copies the molecules with a score above or below a specified threshold to an output directory.
    """
        ref_value = self.trajectory.scores[score_name][target_indx] 
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


    def accuracy_filter(self, input_path, output_path, prop, pjsons, target_indx, threshold=0.1):
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
        
        ref_value = float(self.trajectory.properties[prop][target_indx])
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
            raw_value = get_value(data, keys)
            if raw_value is None:
                print(f"Warning: Property {prop} not found for molecule {mol_name}. Skipping.")
                continue
            value = float(raw_value) 

            if lower_bound <= value <= upper_bound:
                # copy this molecule to the output path
                src_file = ipath / f"{mol_name}.xyz"
                dst_file = opath / f"{mol_name}.xyz"
                if src_file.exists():
                    shutil.copy(src_file, dst_file)

    def accuracy_calculation(self, input_file, output_path, prop, pjsons, target_indx):
        """ Calculate accuracy of molecules based on a property from a file.
        This function calculates the accuracy of molecules based on a specified property
        (e.g., eMBD, HLgap, DIP) from their corresponding JSON files.
        The accuracy is determined by comparing the property value of each molecule
        to a reference value.
        The results are saved as csv in output path.
        """
        # Get the reference value from the trajectory
        ref_value = float(self.trajectory.properties[prop][target_indx])
        # Define the property keys for extraction
        prop_dict = {
            "eMBD": ("SUM", "MBD"),  # nested path
            "HLgap": ("HOMO-LUMO gap",),  # top-level
            "DIP": ("Absolute Total Dipole (eAng):",)  # top-level
        }
        keys = prop_dict[prop]
        # Helper to extract a value of a key from given data
        get_value = lambda data, keys: reduce(
            lambda d, k: d.get(k) if isinstance(d, dict) else None,
            keys if isinstance(keys, (list, tuple)) else (keys,),
            data)
        # Read properties from a json file
        fname = input_file.name
        mol_name = input_file.stem
        jf = pjsons / f"{mol_name}.json"
        with jf.open("r") as json_file:
            data = json.load(json_file)
        # Get value
        value = float(get_value(data, keys))
        error = value - ref_value
        relative_err = error *100 / ref_value
        csv_path = output_path / f"{prop}.csv"
        try:
            # Check if the file exists and read its contents
            if csv_path.exists():
                with csv_path.open('r') as f:
                    content = f.read()
            else:
                content = ""

            # Append only if the entry isn't already there
            if f"{fname}\t" not in content:
                with csv_path.open('a') as f:
                    f.write(f"{fname}\t{value}\t{ref_value}\t{error}\t{relative_err}\n")

        except Exception as e:
            print(f"Failed to write {fname} in {csv_path}: {e}")
        
        