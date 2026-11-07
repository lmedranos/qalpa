import os
import sys
import argparse
import warnings
from pathlib import Path
import numpy as np
#import torch
from ase.io import read, write
from ase.optimize import BFGS

# _COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "components"
# if str(_UTILS_DIR) not in sys.path:
#     sys.path.insert(0, str(_UTILS_DIR))
    
from src.components.mol_calculate import get_calculator

def calc_properties(input_file, prop_folder, calc_name, model_path, device):
    """Calculate properties of a molecule using the specified calculator.

    Parameters
    ----------
    input_file : str
        Path to the input molecule file (.xyz).
    prop_folder : str
        Directory where the calculated properties will be saved.
    calc_name : str
        'DFTB+', 'EquiDTB', or 'FHI-aims'.
    model_path : str
        Path to the model file
    device : str
        Device to run the calculator on ('cpu' or 'cuda').
    """
    # Placeholder for property calculation logic using the specified calculator.
    print(f"Calculating properties for {input_file} using {calc_name} with {calc_name} on {device}.")
        
    file = Path(input_file).expanduser().resolve()
    output_folder = Path(prop_folder).expanduser().resolve() / f"{file.stem}"
    
    # check if output exsist during molecule optimization
    # opt_output_folder = output_folder / f"opt-{file.stem}"
    # if opt_output_folder.exists():
    #     print(f"Optimized structure already exists, skipping property calculation: {opt_output_folder}")
    #     return

    output_folder.mkdir(parents=True, exist_ok=True)

    if file.suffix != '.xyz':
        raise ValueError(f"Unsupported file format: {file.suffix}. Only .xyz files are supported.") 
    
    atoms = read(str(file))
    model_path_resolved = str(Path(model_path).expanduser().resolve()) if model_path else model_path

    # Run in a temp working directory so calculator side-files/logs are isolated there.
    os.chdir(output_folder)
    
    try:
        calc = get_calculator(calc_name, model_path_resolved, device, atoms)
        if calc is None:
            print("No valid calculator available, skipping property calculation.")
            return
        atoms.set_calculator(calc) 
        force_path = output_folder / "forces.txt" 
        forces = atoms.get_forces()
        np.savetxt(force_path, forces, fmt="%.10f")
            
    except Exception as e:
        print(f"Error initializing calculator: {e}")
        return
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate properties of a molecule using DFTB+, FHI-aims, or EquiDTB")
    parser.add_argument("--input",  type=str, required=True, help="Path to the input molecule file (.xyz)")
    parser.add_argument("--output", type=str, required=True, help="Output folder for molecule properties")
    parser.add_argument("--aux_arg", nargs="+", help="Auxiliary arguments to pass to the script (e.g., calculator name, model path, device)")
    args = parser.parse_args()
    
    if args.aux_arg and len(args.aux_arg) >= 2:
        calc_name = args.aux_arg[0]
        model_path = args.aux_arg[1] if args.aux_arg[1].strip().lower() not in {"none", "null", ""} else None
        device = 'cpu' if "cpu" in args.aux_arg else 'cuda' if "cuda" in args.aux_arg else 'cpu'
        calc_properties(args.input, args.output, calc_name, model_path, device)
    else:
        print("Insufficient auxiliary arguments provided. Expected: calculator name, model path, device.")