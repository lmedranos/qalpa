import argparse
import warnings  
from pathlib import Path
import torch
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from ase.calculators.dftb import Dftb

# from so3lr import So3lrCalculator


def get_calculator(calc_name, fpath, device):
    """Instantiate and return the appropriate calculator."""
    if calc_name == 'SO3LR':
        warnings.warn("SO3LR is not yet enabled, skipping.", UserWarning)
        return None
    elif calc_name == 'DFTB+':
        return Dftb(
            label='current_dftb',
            Hamiltonian_SCC='Yes',
            Hamiltonian_MaxSCCIterations='2000',
            Hamiltonian_SCCTolerance='1E-6',
            Hamiltonian_HCorrection_='Damping',
            Hamiltonian_HCorrection_Exponent=4.05,
            Hamiltonian_ThirdOrderFull='Yes',
            Hamiltonian_Dispersion_='MBD',
            Hamiltonian_Dispersion_KGrid='1 1 1',
            Hamiltonian_Dispersion_Beta=0.83,  # PBE
            Hamiltonian_Dispersion_NOmegaGrid=25,
            Hamiltonian_Dispersion_ReferenceSet='ts',
            ParserOptions_ParserVersion='13',
            slako_dir =str(fpath) + "/",
        )
    elif calc_name == 'MACE':
        return MACECalculator(model_paths=fpath, device=device, default_dtype="float32")


def relax_molecule(input, output, calc_name, fpath, device='cpu'):
    """Relax a molecule using the specified calculator.

    Parameters
    ----------
    input : str
        Path to the input molecule file (.xyz).
    output : str
        Directory where the relaxed molecule will be saved.
    calc_name : str
        'SO3LR', 'DFTB+', or 'MACE'.
    fpath : str
        Path to the calculator-specific file (e.g., Slater-Koster files for DFTB+ or model file for MACE).
    device : str
        Device to run the calculator on ('cpu' or 'cuda').
    """
    file = Path(input)
    output_folder = Path(output)
    output_folder.mkdir(parents=True, exist_ok=True)

    if file.suffix != '.xyz':
        raise ValueError(f"Unsupported file format '{file.suffix}'. Only .xyz is supported.")

    output_file = output_folder / f"opt-{file.name}"
    if output_file.exists():
        print(f"Output already exists, skipping: {output_file}")
        return

    calc = get_calculator(calc_name, fpath, device)
    if calc is None:
        print("No valid calculator available, skipping relaxation.")
        return

    atoms = read(str(file))
    atoms.calc = calc

    qn = BFGS(atoms, logfile=None)  # suppress stdout logging
    qn.run(fmax=0.001, steps=4000)

    write(str(output_file), atoms, format='xyz')
    print(f"Relaxed molecule saved as {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Relax a molecule using MACE, DFTB+, or SO3LR.")
    parser.add_argument("--input",  type=str, required=True, help="Path to the input molecule file (.xyz)")
    parser.add_argument("--output", type=str, required=True, help="Output folder for the relaxed molecule")
    parser.add_argument("--calc_name",  type=str, default="DFTB+", help="'SO3LR', 'DFTB+', or 'MACE'")
    parser.add_argument("--fpath", type=str, help="Path to the calculator-specific file")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run the calculator on (default: auto-detect)")
    args = parser.parse_args()

    relax_molecule(args.input, args.output, args.calc_name, args.fpath, args.device)