import sys
import os
from os import path
import argparse
from pathlib import Path
# from os import environ
from parserFHItoJSON import get_data as create_json
import shutil

def postprocess_fhi(ifolder, ofolder):
    """
    Preprocess FHI-aims output file to create a JSON file.
    """
    fname = ifolder.name
    ifile = ifolder / "aims.out"
    # print(f"Input file: {ifile}, Output folder: {ofolder}")
    if ifile.exists():
        os.chdir(ifolder)
        try:
            create_json(str(ifile))
        except Exception as e:
            print(f"Error creating JSON from {ifile}: {e}")
            sys.exit(1)
        os.rename("file.json", ofolder / f"{fname}.json") 

if __name__ == "__main__":
    
    """
    This script creates a JSON output file.
    """
    
    # read input file path and output directory from command line arguments
    parser = argparse.ArgumentParser(description="Create JSON output from FHI-aims output.")
    parser.add_argument("--input", type=str, help="Path to the FHI-aims output file")
    parser.add_argument("--output", type=str, help="Output folder for the JSON file")
    args = parser.parse_args()

    ifolder = Path(args.input)
    ofolder = Path(args.output)
     
    postprocess_fhi(ifolder, ofolder)
      
    # fname = ifolder.name
    # ifile = ifolder / "aims.out"
    # # print(f"Input file: {ifile}, Output folder: {ofolder}")
    # if ifile.exists():
    #     os.chdir(ifolder)
    #     try:
    #         create_json(str(ifile))
    #     except Exception as e:
    #         print(f"Error creating JSON from {ifile}: {e}")
    #         sys.exit(1)
    #     os.rename("file.json", ofolder / f"{fname}.json")        
        
