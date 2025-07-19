import sys
import shutil
import os
from os import path
import argparse
from pathlib import Path
from ase.io import read, write
from pathlib import Path
#from os import environ
# from mpi4py import MPI

if __name__ == "__main__":
    
    """
    This script reads an input file in XYZ format, preprocesses it for FHI-aims.
    """

    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()

    # read input file path and output directory from command line arguments
    parser = argparse.ArgumentParser(description="Create input files for FHI-aims.")
    parser.add_argument("--input", type=str, help="Path to the input XYZ file")
    parser.add_argument("--output", type=str, help="Output folder for the FHI-aims input files")
    args = parser.parse_args()

    ifile = Path(args.input) 
    parent_path = Path(args.output)
    cdata = Path(os.environ["CONFIG"],"nctr1.in")

    os.chdir(parent_path)   
    if ifile.suffix == '.xyz':
        fname = ifile.name
        mol_dir = ifile.stem
        opath = parent_path / mol_dir
        atoms = read(ifile, format='xyz')
        Z = atoms.get_atomic_numbers()
        opath.mkdir(parents=True, exist_ok=True)
        os.chdir(opath)
        write('geometry.in', atoms, scaled=False)
        shutil.copy2(cdata, opath / 'control.in')
        # move input files to output directory
        # shutil.move(ifile, opath / fname)


        
