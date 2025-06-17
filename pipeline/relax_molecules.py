import os
import sys
import time
import socket
import sys
#import ase
from ase.io import read, write
from os import chdir
from os import path
from ase.optimize import BFGS
from mace.calculators import MACECalculator


def relax_molecule(file_path, model_path, output_folder):
    """Relax a molecule using the MACE calculator.
     This function reads a molecule from the specified file, applies the MACE calculator,
    and optimizes its geometry using the BFGS algorithm. The relaxed molecule is then saved
    to the specified output folder with a prefix "opt-". If the relaxed file already exists
    in the output folder, the function will return early to avoid redundant calculations.
    
    parameters:
    ---------------
        - file_path: Path to the input molecule file (e.g., .xyz).
        - model_path: Path to the MACE model.
        - output_folder: Directory where the relaxed molecule will be saved.
   """

    start_time = time.time()
    hostname = socket.gethostname()
    filename = os.path.basename(file_path)
    #print(f"Starting processing {filename} on {hostname}")

    SPcalc = MACECalculator(model_path=model_path, device='cpu', default_dtype="float32")   # should we change device
    fname = file_path.split('/')[-1]
    chdir(output_folder)
    if str(fname[-3:]) == 'xyz':
        if path.exists(output_folder+"opt-"+str(fname)) == True: return
        atoms = read(file_path)
        atoms.set_calculator(SPcalc)
        qn = BFGS(atoms)
        qn.run(fmax=0.001,steps=4000)
        write("opt-"+str(fname), atoms, format='xyz')

    end_time = time.time()
    #print(f"Finished processing {filename} on {hostname} in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    file_path = sys.argv[1]
    output_folder = sys.argv[2]
    model_mace = os.path.join(os.environ["CONFIG"],"MACE_model_swa.model")
    relax_molecule(file_path, model_mace, output_folder)