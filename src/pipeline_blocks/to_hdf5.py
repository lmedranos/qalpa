import sys, math, os
import logging
import argparse
from pathlib import Path
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

def run_to_hdf5(query_dir, prop_dir, appending_flag, h5file_output, context_flag, output_extra_prop, h5file_itr, itr_extra_prop, itr):
    """
    Function to convert filtered molecules into HDF5 format.
    It reads filtered molecules from a directory, extracts properties,
    and writes them into HDF5 files for both the full dataset and the current iteration.
    
    Parameters:
    ---------------
        - query_dir: Directory containing filtered molecules in xyz format.
        - prop_dir: Directory containing JSON files with molecular properties.
        - h5file_output: Path to the output HDF5 file for the full dataset.
        - output_extra_prop: Path to the output HDF5 file for extra properties.
        - h5file_itr: Path to the output HDF5 file for the current iteration.
        - itr_extra_prop: Path to the output HDF5 file for extra properties in iteration files.
        - itr: Current iteration number.
    """
    ## initialize database
    EPBE0_atom = {1:-13.64136507, 6:-1027.59344601, \
                7:-1484.25358313, 8:-2039.74560236, 9: -3139.751675679, 15:-9283.01120606, 16: -10828.726222083, 17: -12516.462339357}

    
    mode = 'a' if appending_flag else 'w'

    with ExitStack() as stack:
        data_output = stack.enter_context(h5py.File(h5file_output, mode)) if h5file_output  else None 
        data_extra_prop = stack.enter_context(h5py.File(output_extra_prop, mode)) if output_extra_prop else None
        data_iter_output = stack.enter_context(h5py.File(h5file_itr, 'w')) if h5file_itr else None
        data_itr_extra_prop = stack.enter_context(h5py.File(itr_extra_prop, 'w')) if itr_extra_prop else None

        files = {
            'data_output': data_output,
            'data_extra_prop': data_extra_prop,
            'data_iter_output': data_iter_output,
            'data_itr_extra_prop': data_itr_extra_prop
        }

        for name, hdf5_file in files.items():
            if hdf5_file is not None:
                group_name =  f"G-active-loop-{itr}"
                grp = hdf5_file.require_group(group_name)
                xyzfiles = [f for f in query_dir.iterdir() if f.is_file() and f.suffix == '.xyz']
                
                for xyzfile in xyzfiles:
                    mol_name = xyzfile.stem
                    if mol_name in grp:
                        print(f"{mol_name} already exists in {name}, skipping...")
                        continue
                    subgrp = grp.create_group(f"{mol_name}")
                    xyz_file_path = query_dir / f"{mol_name}.xyz"       # path to the xyz file if coordinate is not in json file
                    atoms = read(xyz_file_path)
                    Z = atoms.get_atomic_numbers()
                    POS = atoms.get_positions()
                    molecule = {}
                    molecule['atNUM'] = np.array(Z, dtype=np.float64)
                    molecule['atXYZ'] = np.array(POS, dtype=np.float64) 
                          
                    if context_flag:                        # add context properties
                        jf = prop_dir / f"{mol_name}.json"
                             
                        with jf.open("r") as json_file:
                            data = json.load(json_file)

                        if 'extra_prop' in name:           # collect collateral properties for the third hdf5
                            totfor_raw = data.get("Total Forces (eV/Ang):")
                            vdwfor_raw = data.get("van der Waals Forces (eV/Ang):")
                            if totfor_raw is not None:
                                totfor = np.array(totfor_raw, dtype=np.float64)
                                molecule['totFOR'] = totfor
                                if vdwfor_raw is not None:
                                    vdwfor = np.array(vdwfor_raw, dtype=np.float64)
                                    molecule['pbe0FOR'] = totfor - vdwfor

                            HChar_raw = data.get("Hirshfeld Charges:")
                            if HChar_raw is not None:
                                molecule['hCHG'] = np.array(HChar_raw, dtype=np.float64)

                            dip_raw = data.get("Total Dipole (eAng)")
                            if dip_raw is not None:
                                molecule['vDIP'] = np.array(dip_raw, dtype=np.float64)

                            FVol_raw = data.get("Free Atom Volumes:")
                            HVol_raw = data.get("Hirshfeld Volumes:")
                            if FVol_raw is not None and HVol_raw is not None:
                                FVol = np.array(FVol_raw, dtype=np.float64)
                                HVol = np.array(HVol_raw, dtype=np.float64)
                                molecule['hRAT'] = np.divide(HVol, FVol, out=np.zeros_like(HVol), where=FVol!=0)    # define hirshfeld ratios

                        else:                               # collect main properties for the first two hdf5 files
                            ener= np.array([data["Total energy (eV):"]], dtype=np.float64)
                            mbd= np.array([data["SUM"]["MBD"]], dtype=np.float64)
                            pbe0 = ener - mbd
                            EDFTatoms = np.array([sum([ EPBE0_atom[zi] for zi in Z ])], dtype=np.float64)
                            Eat = pbe0 - EDFTatoms

                            molecule['DIP'] = np.array([data["Absolute Total Dipole (eAng):"]], dtype=np.float64)
                            molecule['HLgap'] = np.array([data["HOMO-LUMO gap"]], dtype=np.float64)
                            molecule['eAT'] = Eat
                            molecule['eH'] = np.array([data["HOMO energy"]], dtype=np.float64)
                            molecule['eL'] = np.array([data["LUMO energy"]], dtype=np.float64)
                            molecule['eMBD'] = np.array([data["SUM"]["MBD"]], dtype=np.float64)
                            # molecule['ePBE0'] = pbe0
                            molecule['eTS'] = pbe0
                            # molecule['ePBE0+MBD'] = ener

                            # Optional energy components (absent for DFTB+); skip when not provided
                            ekin_raw = data["SUM"].get("Kinetic Energy")
                            if ekin_raw is not None:
                                molecule['eKIN'] = np.array([ekin_raw], dtype=np.float64)
                            exc_raw = data["SUM"].get("Total XC energy")
                            if exc_raw is not None:
                                molecule['eXC'] = np.array([exc_raw], dtype=np.float64)
                            exx_raw = data["SUM"].get("Hartree-Fock Energy")
                            if exx_raw is not None:
                                molecule['eXX'] = np.array([exx_raw], dtype=np.float64)

                    # Writing properties
                    for key, value in molecule.items():
                        dset = subgrp.create_dataset(key, value.shape, dtype=value.dtype)
                        dset[...] = value
                        # print(f"Written {key} for {mol_name} in {name}")
                        
  
 
def _str2bool(value):
    return str(value).strip().lower() in ("yes", "true", "t", "y", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert filtered molecules to HDF5 format.")
    parser.add_argument("--input_xyz", type=str, required=True, help="Path to the directory containing xyz files.")
    parser.add_argument("--input_prop", type=str, required=True, help="Path to the directory containing JSON files.")
    parser.add_argument("--appending_flag", type=_str2bool, required=True, help="Flag to indicate if the HDF5 file should be appended to (True) or overwritten (False).")
    parser.add_argument("--h5file_output", type=str, required=True, help="Path to the output HDF5 file for the full dataset.")
    parser.add_argument("--context", default=True ,type=_str2bool, help="Flag to indicate if context properties should be added.")
    parser.add_argument("--output_extra_prop", type=str, help="Path to the output HDF5 file for extra properties.")
    parser.add_argument("--h5file_itr", type=str, help="Path to the output HDF5 file for the current iteration.")
    parser.add_argument("--itr_extra_prop", type=str, help="Path to the output HDF5 file for extra properties in iteration files.")
    parser.add_argument("--itr", type=int, required=True, help="Current iteration number.")
    args = parser.parse_args()  
    
    run_to_hdf5(Path(args.input_xyz), Path(args.input_prop), args.appending_flag, args.h5file_output, args.context, args.output_extra_prop, args.h5file_itr, args.itr_extra_prop, args.itr)