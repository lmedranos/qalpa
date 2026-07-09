import os
import argparse
import warnings  
from pathlib import Path
import numpy as np
#import torch
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import MACECalculator
import ase.calculators.mixing
from ase.calculators.dftb import Dftb





def get_calculator(calc_name, model_path, device, atoms):
    """Instantiate and return the appropriate calculator."""
    skfiles_path = Path(os.environ["DFTB_PREFIX"])  # Path to Slater-Koster files for DFTB+
    if calc_name == 'DFTB+':
        return Dftb(
            label='current_dftb',
            atoms=atoms,
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
            slako_dir = str(skfiles_path),
        )
    elif calc_name == 'EquiDTB':
        MLcalc = MACECalculator(model_path=model_path, device=device, default_dtype="float32")
        # Load DFTB calculator with MBD
        DFTBcalc = Dftb(label='current_dftb',
                        atoms=atoms,
                        Hamiltonian_SCC = 'Yes',
                        Hamiltonian_ThirdOrderFull = 'Yes',
                        # Hamiltonian_HCorrection_ = 'Damping',
                        # Hamiltonian_HCorrection_Exponent = 4.05,
                        Hamiltonian_PolynomialRepulsive_ = '',
                        Hamiltonian_PolynomialRepulsive_setForAll = '{Yes}',
                        Hamiltonian_Dispersion_ = 'MBD',
                        Hamiltonian_Dispersion_KGrid = '1 1 1',
                        Hamiltonian_Dispersion_Beta = 0.83, 
                        Hamiltonian_Dispersion_NOmegaGrid = 25,
                        Hamiltonian_Dispersion_ReferenceSet = 'ts',
                        ParserOptions_ParserVersion = '13')
        # Mixing calculators
        return  ase.calculators.mixing.SumCalculator([DFTBcalc,MLcalc])

    else:
        return None

                
