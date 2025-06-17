import sys
import shutil
import os
from os import path
from ase.io import read, write
from os import environ, listdir, mkdir, makedirs, chdir, system, remove
from mpi4py import MPI
from parserFHItoJSON import get_data as create_json

"""
This script reads an input file in XYZ format, processes it with FHI-aims, and creates a JSON output file.
It requires the FHI-aims executable to be available in the specified path.
"""

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

# read input file path and output directory from command line arguments
try:
  ifile = sys.argv[1] ;  opath = sys.argv[2] 
except:
  print("Usage:", sys.argv[0], "infile")
  sys.exit(1)

#aims_cmd = 'srun -n $SLURM_NTASKS /software/rapids/r24.04/FHI-aims/240507-intel-2022a/bin/aims.x > aims.out'
aims_cmd = '/software/rapids/r24.04/FHI-aims/240507-intel-2022a/bin/aims.x > aims.out'
cdir = os.environ["CONFIG"]

idir = os.path.dirname(ifile)
os.chdir(idir)   
if str(ifile[-3:]) == 'xyz' :   
    fname = str(ifile.split('/')[-1][:-4])  # Extract filename without extension
    atoms = read(ifile, format='xyz')
    Z = atoms.get_atomic_numbers()
    if path.exists(opath+fname) == False:
        mkdir(opath+fname)
    os.chdir(opath+fname)
    write('geometry.in', atoms, scaled=False)
    shutil.copy2(cdir+'/nctr1.in', opath+fname+'/control.in')
    system(aims_cmd)
    if path.exists(opath+fname+'/aims.out'):
       create_json(opath+fname+'/aims.out')
       
