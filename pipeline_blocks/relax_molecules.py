import os
import sys
# import time
# import sys
#import ase
from ase.io import read, write
from os import chdir
# from os import path
from ase.optimize import BFGS
from mace.calculators import MACECalculator
import argparse
from pathlib import Path


def relax_molecule(input, output, model):
    """Relax a molecule using the MACE calculator.
     This function reads a molecule from the specified file, applies the MACE calculator,
    and optimizes its geometry using the BFGS algorithm. The relaxed molecule is then saved
    to the specified output folder with a prefix "opt-". If the relaxed file already exists
    in the output folder, the function will return early to avoid redundant calculations.
    
    parameters:
    ---------------
        - input: Path to the input molecule file (e.g., .xyz).
        - model: Path to the MACE model.
        - output: Directory where the relaxed molecule will be saved.
   """
    file = Path(input)
    output_folder = Path(output)
    model_path = model
    
    SPcalc = MACECalculator(model_paths=model_path, device='cpu', default_dtype="float32")   # should we change device
    fname = file.name
    if file.suffix == '.xyz':
        output_file = output_folder / f"opt-{fname}"
        if output_file.exists():
            return
        atoms = read(str(file))
        #atoms.set_calculator(SPcalc)
        atoms.calc = SPcalc
        qn = BFGS(atoms)
        qn.run(fmax=0.001,steps=4000)
        write(str(output_file), atoms, format='xyz')
        print(f"Relaxed molecule saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relax a molecule using MACE.")
    parser.add_argument("--input", type=str, help="Path to the input molecule file")
    parser.add_argument("--output", type=str, help="Output folder for the relaxed molecule")
    try:
        config_dir = os.environ["CONFIG"]
        default_model_path = Path(config_dir) / "MACE_model_swa.model"
        if not default_model_path.exists():
            print(f"Error: Default model not found at {default_model_path}", file=sys.stderr)
            sys.exit(1)
        default_model_path = str(default_model_path)  # Convert after check
    except KeyError:
        print("Error: CONFIG environment variable not set.", file=sys.stderr)
        sys.exit(1)
    parser.add_argument("--model", type=str, default=default_model_path, help="Path to the MACE model")
    args = parser.parse_args()

    relax_molecule(args.input, args.output, args.model)