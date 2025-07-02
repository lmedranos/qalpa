import sys, math, os
import logging
import numpy as np
from ase.io import read, write
from os import environ, listdir, mkdir, chdir
from ase.units import Hartree, Bohr
from os import path
from ase import Atoms
import json
from ase.io import read
import h5py

## initialize database
EPBE0_atom = {1:-13.64136507, 6:-1027.59344601, \
              7:-1484.25358313, 8:-2039.74560236, 9: -3139.751675679, 15:-9283.01120606, 16: -10828.726222083, 17: -12516.462339357}

filtered_dir = sys.argv[1]
h5file_path = sys.argv[2]            # output file path 
itr = 0
folders = sorted(listdir(filtered_dir))

with h5py.File("/data/horse/ws/miha493e-activediffusion_hanna/datasets/new.hdf5", 'w') as h5file:

    for fdir in folders:#[:1]:

        print(fdir)
        grp = h5file.create_group(f"{fdir}_{itr}")
        molecule = {}

        #folders2 = sorted(listdir(xyz_pre+fdir))
        folder_path = os.path.join(filtered_dir, fdir)
        jfils = [f for f in listdir(folder_path) if f.endswith('.json')]  #As folder have non json files

        for confid in jfils:

            with open(filtered_dir+'/'+str(fdir)+'/'+str(confid)) as json_file:
                data = json.load(json_file)

            sysname = confid[:-5]

            subgrp = grp.create_group(sysname)
        #saving DFT data

            # w = len(data["Coordinates"])
            # ftmp = open('tmp-'+str(sysname)+'.xyz', 'w')  # open file for writing
            # ftmp.write(str(w) + "\n")
            # ftmp.write("{: >24}".format('leo') + "\n")
            # for p1 in data["Coordinates"]:
            #     ftmp.write("{: >24} {: >24} {: >24} {: >24}".format(*p1) + "\n")
            # ftmp.close()
            #atoms = read('tmp-'+str(sysname)+'.xyz')    # define atoms
            xyz_file_path = os.path.join(folder_path, sysname + '.xyz')       # path to the xyz file if coordinate is not in json file
            atoms = read(xyz_file_path)
            Z = atoms.get_atomic_numbers()
            POS = atoms.get_positions()

            #os.remove('tmp-'+str(sysname)+'.xyz')

            ener=float(data["Total energy (eV):"])
            mbd=float(data["SUM"]["MBD"])
            pbe0 = float(ener) - float(mbd)
            print("EPBE0_atom keys:", EPBE0_atom.keys())
            print("Z values:", Z)
            print([type(zi) for zi in Z])
            EDFTatoms = sum([ EPBE0_atom[zi] for zi in Z ])
            Eat = float(pbe0) - float(EDFTatoms)

            totfor = np.array(data["Total Forces (eV/Ang):"])
            vdwfor = np.array(data["van der Waals Forces (eV/Ang):"])
            pbe0for = totfor - vdwfor

        #Hirshfeld quantities
            HChar = np.array(data["Hirshfeld Charges:"])

        #DIPOLE moment
#            dipole = float(data["Absolute Total Dipole (eAng):"])
#            dipcomp = np.array(data["Total Dipole (eAng)"]) 

            FVol = data["Free Atom Volumes:"]
            HVol = data["Hirshfeld Volumes:"]
            hvr = np.divide(HVol,FVol)    # define hirshfeld ratios

            molecule['atXYZ'] = np.array(POS)
            molecule['atNUM'] = np.array(Z)

            molecule['ePBE0+MBD'] = np.array([float(ener)])
            molecule['ePBE0'] = np.array([float(pbe0)])
            molecule['totFOR'] = totfor
            molecule['pbe0FOR'] = pbe0for

            molecule['hCHG'] = HChar

#            molecule['DIP'] = np.array([float(dipole)])
#            molecule['vDIP'] = dipcomp
            molecule['hRAT'] = hvr

            for key, value in molecule.items():
                dset = subgrp.create_dataset(key, value.shape, dtype=value.dtype)
                dset[...] = value
