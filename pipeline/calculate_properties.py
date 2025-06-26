import sys
import shutil
import os
from os import path
import subprocess
from ase.io import read, write
from os import environ
from mpi4py import MPI
from parserFHItoJSON import get_data as create_json

if __name__ == "__main__":
    """
    This script reads an input file in XYZ format, processes it with FHI-aims, and creates a JSON output file.
    It requires the FHI-aims executable to be available in the specified path.
    """

    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()

    # read input file path and output directory from command line arguments
    ifile = sys.argv[1]  , cdata = sys.argv[2]
    print(f"Processing file: {ifile} with control data: {cdata}")
    #aims_cmd = 'srun -n $SLURM_NTASKS /software/rapids/r24.04/FHI-aims/240507-intel-2022a/bin/aims.x > aims.out'
    #aims_cmd='/software/rapids/r24.04/FHI-aims/240507-intel-2022a/bin/aims.x > aims.out'
    aims_cmd='/software/rapids/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x > aims.out' 

    #cdata = os.path.join(os.environ["CONFIG"], "nctr1.in")

    idir = os.path.dirname(ifile)
    os.chdir(idir)   
    if ifile.lower().endswith('.xyz'):
        fname = str(ifile.split('/')[-1][:-4])  # Extract filename without extension
        opath = idir/fname
        atoms = read(ifile, format='xyz')
        Z = atoms.get_atomic_numbers()
        os.makedirs(opath, exist_ok=True)
        os.chdir(opath)
        write('geometry.in', atoms, scaled=False)
        shutil.copy2(cdata, opath+'/control.in')
        subprocess.run(aims_cmd, shell=True, check=True)
        if path.exists(opath+'/aims.out'):
          create_json(opath+'/aims.out')
        # move input files to output directory
        shutil.move(ifile, os.path.join(opath, f'{fname}.xyz'))
        
