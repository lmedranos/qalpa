import os
import sys
import argparse
import warnings  
from pathlib import Path
import numpy as np
#import torch
from ase.io import read, write
from ase.optimize import BFGS

    
from utils.mol_calculate import get_calculator

def relax_molecule(input, output, calc_name, model_path=None, device='cpu', relaxed_forces=False, unrelaxed_forces=False, keep_temp=False):
    """Relax a molecule using the specified calculator.

    Parameters
    ----------
    input : str
        Path to the input molecule file (.xyz).
    output : str
        Directory where the relaxed molecule will be saved.
    calc_name : str
        'SO3LR', 'DFTB+', 'EquiDTB', or 'MACE'.
    model_path : str
        Path to the model file (e.g., model file for MACE).
    device : str
        Device to run the calculator on ('cpu' or 'cuda').
    relaxed_forces : bool
        Whether to calculate and save forces after relaxation (default: False).
    unrelaxed_forces : bool
        Whether to calculate and save forces before relaxation (default: False).
    keep_temp : bool
        Whether to keep temporary files (default: False).
    """
    file = Path(input).expanduser().resolve()
    output_folder = Path(output).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    if file.suffix != '.xyz':
        raise ValueError(f"Unsupported file format: {file.suffix}. Only .xyz files are supported.") 
    
    output_file = output_folder / f"opt-{file.name}"
    if output_file.exists():
        print(f"Output already exists, skipping: {output_file}")
        return

    temp_dir = output_folder / f"temp-{file.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(str(file))
    model_path_resolved = str(Path(model_path).expanduser().resolve()) if model_path else model_path

    # Run in a temp working directory so calculator side-files/logs are isolated there.
    original_cwd = Path.cwd()
    os.chdir(temp_dir)

    try:
        calc = get_calculator(calc_name, model_path_resolved, device, atoms)
        if calc is None:
            print("No valid calculator available, skipping relaxation.")
            return

        atoms.calc = calc

        if unrelaxed_forces:
            (output_folder.parent / "force_calc" / "unrelaxed").mkdir(parents=True, exist_ok=True)
            unrelaxed_path = output_folder.parent / "force_calc" / "unrelaxed" 
            forces_before = atoms.get_forces()
            energy_before = atoms.get_potential_energy()
            np.savetxt(unrelaxed_path / f"forces_{file.stem}.txt", forces_before, fmt="%.10f")
            write(unrelaxed_path / f"pos_{file.stem}.xyz", atoms, format='xyz')
            atoms.arrays['forces'] = forces_before
            write(unrelaxed_path / f"{file.stem}.extxyz", atoms, format='extxyz')  # includes forces, energy, positions
            del atoms.arrays['forces']

        qn = BFGS(atoms, logfile=None)  # suppress optimizer stdout logging
        qn.run(fmax=0.001, steps=4000)

        if relaxed_forces:
            (output_folder.parent / "force_calc" / "relaxed").mkdir(parents=True, exist_ok=True)
            relaxed_path = output_folder.parent / "force_calc" / "relaxed" 
            forces_after = atoms.get_forces()
            energy_after = atoms.get_potential_energy()
            np.savetxt(relaxed_path / f"forces_opt-{file.stem}.txt", forces_after, fmt="%.10f")
            write(relaxed_path / f"pos_opt-{file.stem}.xyz", atoms, format='xyz')
            atoms.arrays['forces'] = forces_after
            write(relaxed_path / f"opt-{file.stem}.extxyz", atoms, format='extxyz')  # includes forces, energy, positions
            del atoms.arrays['forces']

    finally:
        os.chdir(original_cwd)

    write(str(output_file), atoms, format='xyz')
    print(f"Relaxed molecule saved as {output_file}")

    if not keep_temp:
        for item in temp_dir.iterdir():
            if item.is_file():
                item.unlink()
        temp_dir.rmdir()
    


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Relax a molecule using MACE, DFTB+, or SO3LR.")
    parser.add_argument("--input",  type=str, required=True, help="Path to the input molecule file (.xyz)")
    parser.add_argument("--output", type=str, required=True, help="Output folder for the relaxed molecule")
    parser.add_argument("--aux_arg", nargs="+", help="Auxiliary arguments to pass to the script (e.g., calculator name, model path, device)")
    args = parser.parse_args()
    
    if args.aux_arg and len(args.aux_arg) >= 2:
        calc_name = args.aux_arg[0]
        model_path = args.aux_arg[1] if args.aux_arg[1].strip().lower() not in {"none", "null", ""} else None
        relaxed_forces = True if "relaxed_forces" in args.aux_arg else False
        unrelaxed_forces = True if "unrelaxed_forces" in args.aux_arg else False
        keep_temp = True if "keep_temp" in args.aux_arg else False
        device = 'cpu' if "cpu" in args.aux_arg else 'cuda' if "cuda" in args.aux_arg else 'cpu'
        relax_molecule(args.input, args.output, calc_name, model_path, device, relaxed_forces, unrelaxed_forces, keep_temp)
    else:
        print("Insufficient auxiliary arguments provided. Expected: calculator name, model path, device.")