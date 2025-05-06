import sys
import shutil
import os
from os import path
from ase.io import read, write
from os import environ, listdir, mkdir, chdir, system, remove

try:
  ifile = sys.argv[1] ;  ofile = sys.argv[2] 
except:
  print("Usage:", sys.argv[0], "infile")
  sys.exit(1)

aims_cmd = 'srun -n $SLURM_NTASKS /software/rapids/r24.04/FHI-aims/240507-intel-2022a/bin/aims.x > aims.out'
wdir = '/project/scratch/p200243/Medrano/julian/qm7x-aqm-neq6/code/fhi/ps1/'

xyz_pre = ifile
os.chdir(xyz_pre)
folders = sorted(listdir('.'))

for fname in folders:
    if str(fname[-3:]) == 'xyz':

        fdir = str(fname[:-4])

        atoms = read(xyz_pre+fdir+'.xyz', format='xyz')

        Z = atoms.get_atomic_numbers()

        print(fdir)
        if path.exists(ofile+fdir) == False:
            mkdir(ofile+fdir)
        os.chdir(ofile+fdir)

        write('geometry.in', atoms, scaled=False)

        shutil.copy2(wdir+'nctr1.in', ofile+fdir+'/control.in')
        system(aims_cmd)
        chdir(ofile)
