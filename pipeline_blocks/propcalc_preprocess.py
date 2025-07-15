import sys
import shutil
import os
from os import path
from ase.io import read, write
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
    ifile = sys.argv[1]  
    cdata = sys.argv[2]
    #env = os.environ.copy()

    idir = os.path.dirname(ifile)
    os.chdir(idir)   
    if ifile.lower().endswith('.xyz'):
        fname = str(ifile.split('/')[-1][:-4])  # Extract filename without extension
        opath = os.path.join(idir, fname)
        atoms = read(ifile, format='xyz')
        Z = atoms.get_atomic_numbers()
        os.makedirs(opath, exist_ok=True)
        os.chdir(opath)
        write('geometry.in', atoms, scaled=False)
        shutil.copy2(cdata, opath+'/control.in')
        # move input files to output directory
        shutil.move(ifile, os.path.join(opath, f'{fname}.xyz'))


        
