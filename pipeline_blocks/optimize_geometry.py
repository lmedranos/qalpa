import argparse
from pathlib import Path
import torch
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import MACECalculator
# from so3lr import So3lrCalculator


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
    mydevice = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    file = Path(input)
    output_folder = Path(output)
    model = model
    
    if model =='SO3LR':
        # calc = So3lrCalculator(calculate_stress=False, dtype='float32', device= mydevice)
        pass
    else:
        calc = MACECalculator(model_paths=model, device='cpu', default_dtype="float32")   # should we change device
    
    if file.suffix == '.xyz':
        fname = file.name
        output_file = output_folder / f"opt-{fname}"
        if output_file.exists():
            return
        atoms = read(str(file))
        atoms.calc = calc
        qn = BFGS(atoms)
        qn.run(fmax=0.001,steps=4000)
        write(str(output_file), atoms, format='xyz')
        print(f"Relaxed molecule saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relax a molecule using MACE.")
    parser.add_argument("--input", type=str, help="Path to the input molecule file")
    parser.add_argument("--output", type=str, help="Output folder for the relaxed molecule") 
    parser.add_argument("--model", type=str, default="SO3LR", help="Path to the MACE model")
    args = parser.parse_args()

    relax_molecule(args.input, args.output, args.model)