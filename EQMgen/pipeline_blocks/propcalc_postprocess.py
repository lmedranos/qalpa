import sys
import os
from os import path
import argparse
from pathlib import Path
# from os import environ
from .parserFHItoJSON import get_data as create_FHI_json
from .parserEquiDTBtoJSON import get_data as create_EquiDTB_json
import shutil
from collections import deque

def valid_fhi_out(fhi_file):
    if not fhi_file.exists():
        return False
    marker=" Have a nice day."
    # Search only the last `scope` lines
    with fhi_file.open('r') as f:
        last_lines = deque(f, maxlen=20)
    return any(marker in line for line in last_lines)
    
# This function in going to be deprecated
def postprocess_fhi(input, ofolder):
    """
    Postprocess FHI-aims output file to create a JSON file.
    """
    if input.is_dir():
        # passing fhi-aim calculation directory, will be deprecated
        fname = input.name
        ifile = input / "aims.out"
    else:
        # passing query files
        fname = input.stem
        ifile = ofolder/ fname / "aims.out"
        
    jfile = ofolder / f"{fname}.json"
    if (valid_fhi_out(ifile)) and (not jfile.exists()) :
        try:
            create_FHI_json(str(ifile), str(jfile)) 
        except Exception as e:
            print(f"Error creating JSON from {ifile}: {e}")

def postprocess_equidtb(input, ofolder):
    """
    Postprocess EquiDTB output file to create a JSON file.
    """
    fname = input.stem
    ifolder = ofolder/ fname
    jfile = ofolder / f"{fname}.json"
    if not jfile.exists():
        try:
            create_EquiDTB_json(str(ifolder), str(jfile))
        except Exception as e:
            print(f"Error creating JSON from {input}: {e}")

    
    
def prop_postprocess(input, ofolder, prop_calc):
    if prop_calc == "FHI-aims":
        postprocess_fhi(input, ofolder)
    elif prop_calc == "EquiDTB":
        postprocess_equidtb(input, ofolder)
    else:
        print(f"Unsupported property calculator for postprocessing: {prop_calc}. Skipping postprocessing.")

if __name__ == "__main__":
    
    """
    This script creates a JSON output file.
    """
    
    # read input file path and output directory from command line arguments
    parser = argparse.ArgumentParser(description="Create JSON output from FHI-aims output.")
    parser.add_argument("--input", type=str, help="Path to the FHI-aims/EquiDTB output file")
    parser.add_argument("--output", type=str, help="Output folder for the JSON file")
    parser.add_argument("--aux_arg", nargs="+", help="Auxiliary arguments to pass to the script (e.g., calculator name)")
    args = parser.parse_args()

    ifolder = Path(args.input)
    ofolder = Path(args.output)
    prop_calc = args.aux_arg[0] if args.aux_arg else "FHI-aims"  # default to FHI-aims if not specified
     
    prop_postprocess(ifolder, ofolder, prop_calc)
      
