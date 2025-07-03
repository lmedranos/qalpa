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
    ifile = sys.argv[1]  
    cdata = sys.argv[2]
    env = os.environ.copy()
    #aims_cmd = 'srun -n $SLURM_NTASKS /software/rapids/r24.04/FHI-aims/240507-intel-2022a/bin/aims.x > aims.out'
    #aims_cmd = '/software/rapids/r24.04/FHI-aims/240507-intel-2022a/bin/aims.x > aims.out'
    #aims_cmd = '/software/rapids/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x > aims.out' 
    #aims_cmd = 'srun -n $SLURM_NTASKS /software/rapids/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x'

    #cdata = os.path.join(os.environ["CONFIG"], "nctr1.in")

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

        # Run FHI-aims
        # subprocess.run(aims_cmd, shell=True, check=True)

        # aims_cmd = ['srun', '/software/rapids/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x']
        # with open('aims.out', 'w') as outfile:
        #     subprocess.run(aims_cmd, env=env, stdout=outfile, stderr=subprocess.STDOUT, check=True)
        
        # sh.Command(aims_cmd)(_out="aims.out")
        aims_cmd = "/software/rapids/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x",

        # fd = os.open("aims.out", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        # os.dup2(fd, 1)  # Redirect stdout
        # os.dup2(fd, 2)  # Redirect stderr
        # os.close(fd)
        # os.execvp(aims_cmd, [aims_cmd])
        os.system(f'srun -n $SLURM_NTASKS /software/rome/r24.04/FHI-aims/240507-intel-2023b/bin/aims.x > aims.out')

        if path.exists(opath+'/aims.out'):
          create_json(opath+'/aims.out')
          # rename the file from file.json to fnam.json
          os.rename("file.json", f"{fname}.json")
        # move input files to output directory
        shutil.move(ifile, os.path.join(opath, f'{fname}.xyz'))
        
