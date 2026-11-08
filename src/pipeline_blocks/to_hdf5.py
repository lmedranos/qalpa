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
import shutil
from contextlib import ExitStack

from src.components.dataset_labels import extract_labels
from src.components.candidates import canonical_id
from src.components.dataset_labels import valid_qm_result


def run_to_hdf5(query_dir, prop_dir, appending_flag, h5file_output, context_flag, output_extra_prop,
                h5file_itr, itr_extra_prop, itr, al_meta=None, al_props=None, dataset=None):
    """Append molecules to the HDF5 files; the cumulative file is backed up and restored if the write fails."""
    backup = Path(str(h5file_output) + ".bak") if (appending_flag and h5file_output) else None
    if backup is not None and backup.exists():
        # A previous run died mid-write: the backup is the last consistent copy.
        print(f"Restoring {h5file_output} from interrupted run backup {backup}")
        shutil.copy2(backup, h5file_output)
    elif backup is not None and Path(h5file_output).exists():
        shutil.copy2(h5file_output, backup)
    else:
        backup = None
    try:
        _write_hdf5(query_dir, prop_dir, appending_flag, h5file_output, context_flag, output_extra_prop,
                    h5file_itr, itr_extra_prop, itr, al_meta, al_props, dataset)
    except BaseException:
        if backup is not None and backup.exists():
            shutil.copy2(backup, h5file_output)
        raise
    if backup is not None and backup.exists():
        backup.unlink()


def _write_hdf5(query_dir, prop_dir, appending_flag, h5file_output, context_flag, output_extra_prop,
                h5file_itr, itr_extra_prop, itr, al_meta=None, al_props=None, dataset=None):
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
                    xyz_file_path = query_dir / f"{mol_name}.xyz"       # path to the xyz file if coordinate is not in json file
                    if context_flag:
                        ok, reason = valid_qm_result(prop_dir / f"{mol_name}.json", xyz_file_path, [], dataset=dataset)
                        if not ok:
                            print(f"Skipping {mol_name} in {name}: {reason}")
                            continue
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

                        label_group = "extra_hdf5_labels" if "extra_prop" in name else "hdf5_labels"
                        molecule.update(extract_labels(data, Z, dataset=dataset, group=label_group))

                    if al_meta is not None and 'extra_prop' not in name:
                        meta = al_meta.get(canonical_id(mol_name))
                        if meta is not None:
                            molecule['al_iteration'] = np.array([meta['iteration']], dtype=np.int64)
                            molecule['al_target_id'] = np.array([meta['target_id']], dtype=np.int64)
                            molecule['al_requested'] = np.array(meta['requested'], dtype=np.float64)
                            molecule['al_hit'] = np.array([meta['hit']], dtype=np.int8)

                    # Writing properties: the group is created only now and removed if a dataset fails
                    subgrp = grp.create_group(f"{mol_name}")
                    try:
                        if al_props and 'al_requested' in molecule:
                            subgrp.attrs['al_props'] = ','.join(al_props)
                        for key, value in molecule.items():
                            dset = subgrp.create_dataset(key, value.shape, dtype=value.dtype)
                            dset[...] = value
                    except Exception:
                        del grp[mol_name]
                        raise
                        
  
 
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