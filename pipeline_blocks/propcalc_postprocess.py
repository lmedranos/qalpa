import sys
import os
from os import path
# from os import environ
from parserFHItoJSON import get_data as create_json

if __name__ == "__main__":
    
    """
    This script creates a JSON output file.
    """
    
    # read input file path and output directory from command line arguments
    ifile = sys.argv[1]  
    cdata = sys.argv[2]
    # env = os.environ.copy()

    idir = os.path.dirname(ifile)
    if ifile.lower().endswith('.xyz'):
        fname = str(ifile.split('/')[-1][:-4])  # Extract filename without extension
        opath = os.path.join(idir, fname)
        os.chdir(opath)
        if path.exists(opath+'/aims.out'):
          create_json(opath+'/aims.out')
          # rename the file from file.json to fnam.json
          os.rename("file.json", f"{fname}.json")
        
        
