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
from contextlib import ExitStack

## initialize database
EPBE0_atom = {1:-13.64136507, 6:-1027.59344601, \
              7:-1484.25358313, 8:-2039.74560236, 9: -3139.751675679, 15:-9283.01120606, 16: -10828.726222083, 17: -12516.462339357}

filtered_dir = sys.argv[1]           # input directory with filtered molecules
h5file_output = sys.argv[3]          # output file path of full dataset
output_extra_prop = sys.argv[4]      # output file path of extra properties
h5file_itr = sys.argv[5]             # output file path of current iteration
itr_extra_prop = sys.argv[6]         # output file path of extra properties in iteration files
itr = sys.argv[7]

folders = sorted(listdir(filtered_dir))

mode = 'a' if (sys.argv[2]=='True') else 'w'

with ExitStack() as stack:
    data_output = stack.enter_context(h5py.File(h5file_output, mode)) if (h5file_output != 'None') else None 
    data_extra_prop = stack.enter_context(h5py.File(output_extra_prop, mode)) if (output_extra_prop!= 'None') else None
    data_iter_output = stack.enter_context(h5py.File(h5file_itr, 'w')) if (h5file_itr!= 'None') else None
    data_itr_extra_prop = stack.enter_context(h5py.File(itr_extra_prop, 'w')) if (itr_extra_prop!= 'None') else None

    files = {
        'data_output': data_output,
        'data_extra_prop': data_extra_prop,
        'data_iter_output': data_iter_output,
        'data_itr_extra_prop': data_itr_extra_prop
    }

    for name, hdf5_file in files.items():
        if hdf5_file is not None:
            grp = hdf5_file.create_group(f"G-{itr}") 
            for fdir in folders:

                #grp = hdf5_file.create_group(f"{fdir}_itr_{itr}")
                molecule = {}

                # folders2 = sorted(listdir(xyz_pre+fdir))
                folder_path = os.path.join(filtered_dir, fdir)
                jfils = [f for f in listdir(folder_path) if f.endswith('.json')]  #As folder have non json files

                for confid in jfils:

                    with open(filtered_dir+'/'+str(fdir)+'/'+str(confid)) as json_file:
                        data = json.load(json_file)

                    sysname = confid[:-5]

                    subgrp = grp.create_group(sysname)
    
                    xyz_file_path = os.path.join(folder_path, sysname + '.xyz')       # path to the xyz file if coordinate is not in json file
                    atoms = read(xyz_file_path)
                    Z = atoms.get_atomic_numbers()
                    POS = atoms.get_positions()

                    # os.remove('tmp-'+str(sysname)+'.xyz')

                    ener= np.array([data["Total energy (eV):"]], dtype=np.float64)
                    mbd= np.array([data["SUM"]["MBD"]], dtype=np.float64)
                    pbe0 = ener - mbd
                    EDFTatoms = np.array([sum([ EPBE0_atom[zi] for zi in Z ])], dtype=np.float64)
                    Eat = pbe0 - EDFTatoms

                    totfor = np.array(data["Total Forces (eV/Ang):"], dtype=np.float64)
                    vdwfor = np.array(data["van der Waals Forces (eV/Ang):"], dtype=np.float64)
                    pbe0for = totfor - vdwfor

                    HChar = np.array(data["Hirshfeld Charges:"], dtype=np.float64)
                    dipcomp = np.array(data["Total Dipole (eAng)"], dtype=np.float64) 

                    FVol = np.array(data["Free Atom Volumes:"], dtype=np.float64)
                    HVol = np.array(data["Hirshfeld Volumes:"], dtype=np.float64)
                    hvr = np.divide(HVol, FVol, out=np.zeros_like(HVol), where=FVol!=0)    # define hirshfeld ratios
                    
                    if 'extra_prop' in name:           # collect collateral properties for the third hdf5
                        molecule['totFOR'] = totfor
                        molecule['pbe0FOR'] = pbe0for
                        molecule['hCHG'] = HChar
                        molecule['vDIP'] = dipcomp
                        molecule['hRAT'] = hvr

                    else:                               # collect main properties for the first two hdf5 files
                        molecule['DIP'] = np.array([data["Absolute Total Dipole (eAng):"]], dtype=np.float64)
                        molecule['HLgap'] = np.array([data["HOMO-LUMO gap"]], dtype=np.float64)
                        molecule['atNUM'] = np.array(Z, dtype=np.float64)
                        molecule['atXYZ'] = np.array(POS, dtype=np.int64)
                        molecule['eAT'] = (pbe0) - (EDFTatoms)
                        molecule['eH'] = np.array([data["HOMO energy"]], dtype=np.float64)
                        molecule['eKIN'] = np.array([data["SUM"]["Kinetic Energy"]], dtype=np.float64)
                        molecule['eL'] = np.array([data["LUMO energy"]], dtype=np.float64)
                        molecule['eMBD'] = np.array([data["SUM"]["MBD"]], dtype=np.float64)
                        molecule['ePBE0'] = pbe0
                        molecule['ePBE0+MBD'] = ener     
                        molecule['eXC'] = np.array([data["SUM"]["Total XC energy"]], dtype=np.float64)
                        molecule['eXX_eHF'] = np.array([data["SUM"]["Hartree-Fock Energy"]], dtype=np.float64)

                    

                    # Writing properties
                    for key, value in molecule.items():
                        dset = subgrp.create_dataset(key, value.shape, dtype=value.dtype)
                        dset[...] = value
