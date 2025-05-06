import sys, math
import logging
import numpy as np
#import ase
from ase.io import read, write
from ase.db import connect
from os import environ, listdir, mkdir, chdir
import torch
import torch.nn as nn
from ase.calculators.dftb import Dftb
from ase.units import Hartree, Bohr
from ase.constraints import FixAtoms, FixInternals
from os import path

from ase.optimize import QuasiNewton, BFGS
import ase.calculators.mixing
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.units import Hartree, Bohr, kcal, fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mace.calculators import MACECalculator

SPcalc = MACECalculator(model_path=sys.argv[1], device='cpu', default_dtype="float32")

xyz_pre = sys.argv[2]
odir = sys.argv[3]

#computing predicted property
logging.info("get predicted property")
flist = listdir(xyz_pre)
flist2 = sorted(flist)

chdir(odir)

for iFile, fname in enumerate(flist2):
    if str(fname[-3:]) == 'xyz':

        sysname = fname[:-4]
        if path.exists(odir+"opt-"+str(fname)) == True: continue
        print(sysname)

        atoms = read(xyz_pre+fname)

        atoms.set_calculator(SPcalc)

        qn = BFGS(atoms)
        qn.run(fmax=0.001,steps=4000)

        write("opt-"+str(fname), atoms, format='xyz')

