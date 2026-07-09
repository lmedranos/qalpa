from sys  import argv
from os.path import isfile, join
from os import getcwd
import re
import json
import numpy as np
import os

from argparse import ArgumentParser

import numpy as np

from ase.io import read
from ase.atoms import Atoms
from ase.data import atomic_numbers
from ase.calculators.dftb import Dftb
from ase.units import Hartree, Bohr

"""
This script reads the output file from EquiDTB files and extracts various properties such as coordinates, energy, components, dipole moment, Hirshfeld charges, volumes, forces, etc.
It then prints these properties and saves them in a JSON file.
"""

def get_data(ifolder, jfile='file.json'):
    
    try:
    # Extract properties
        # band energies
        eig = get_band_energies(ifile_path=join(ifolder, 'band.out'))
        # Mulliken charges
        charges = get_atom_charges(ifile_path=join(ifolder, 'detailed.out'))
        # Total forces (combined DFTB+ + ML) written by prop_calc as forces.txt
        F = np.loadtxt(join(ifolder, 'forces.txt')).reshape(-1, 3).tolist()

        # Read DFTB+ output file
        ifile = open(join(ifolder, 'detailed.out'), 'r')
        lines = ifile.readlines()
        ifile.close()
        # Parse line by line for quantities of interest
        for line in lines:
            if line.strip().startswith('Fermi level:'):
                FermiEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Band energy:'):
                BandEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Input / Output electrons (q):'):
                NumElec = float(line.split()[5])
            if line.strip().startswith('Energy H0:'):
                h0Ene = float(line.split()[2]) * Hartree
            if line.strip().startswith('Energy SCC:'):
                sccEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Energy 3rd:'):
                thirdEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Repulsive energy:'):
                repEne = float(line.split()[2]) * Hartree
            if line.strip().startswith('Dispersion energy:'):
                mbdEne = float(line.split()[2]) * Hartree
            if "Dipole moment:" in line:
                lsp = line.split()
                if lsp[-1] == 'au':
                    # converted from a.u. (e × Bohr) to e × Å using Bohr constant
                    DipMom = np.array([ float(lsp[-4]), float(lsp[-3]), float(lsp[-2])])*Bohr
                break

        # --- Derived quantities (physical meaning of the DFTB+ outputs) ---
        # DFTB+ total energy is the sum of its energy contributions:
        #   E0 (H0) + E_SCC + E_3rd + E_repulsive + E_dispersion
        E = float(h0Ene + sccEne + thirdEne + repEne + mbdEne)

        # Kohn-Sham eigenvalues around the gap. get_band_energies() returns
        # [flag, sorted eigenvalues]; with iorbs=4 the set is the 4 highest
        # occupied followed by the 4 lowest virtual orbitals, sorted ascending,
        # so index 3 is the HOMO and index 4 the LUMO.
        flag = int(eig[0])
        kse2 = [float(e) for e in eig[1:]]
        if flag == 8 and len(kse2) >= 8:
            eHOMO = kse2[3]
            eLUMO = kse2[4]
            HLgap = eLUMO - eHOMO
        else:
            eHOMO = eLUMO = HLgap = 0.0

        # Total dipole vector and its magnitude
        d = [float(c) for c in DipMom]
        ad = float(np.linalg.norm(DipMom))

        # collect into a JSON structure (keys kept identical to the FHI-aims
       
        output={}
        output["Total energy (eV):"]=E
        
        output["SUM"]={}
        output["SUM"]["Sum of eigenvaluess"]=BandEne   # band energy = sum of occupied KS eigenvalues
        output["SUM"]["MBD"]=mbdEne                     # dispersion (vdW) energy
        output["SUM"]["Kinetic Energy"]= None             # [main] needs fill -> to_hdf5 molecule['eKIN']
        output["SUM"]["Electrostatic Energy"]=sccEne    # SCC = 2nd-order electrostatic charge-fluctuation energy
        output["SUM"]["Hartree-Fock Energy"]= None         # [main] needs fill -> to_hdf5 molecule['eXX']
        output["SUM"]["Total XC energy"]= None             # [main] needs fill -> to_hdf5 molecule['eXC']

        output["Hirshfeld Charges:"]=charges            # DFTB+ Mulliken charges (population analogue)
        output["Free Atom Volumes:"]= None              # [extra_prop] needs fill -> to_hdf5 molecule['hRAT']
        output["Hirshfeld Volumes:"]= None                # [extra_prop] needs fill -> to_hdf5 molecule['hRAT']
        output["Kohn-Sham Eigenvalues:"]=kse2
        output["HOMO energy"]=eHOMO
        output["LUMO energy"]=eLUMO
        output["HOMO-LUMO gap"]=HLgap
        output["Total Dipole (eAng)"]=d
        output["Absolute Total Dipole (eAng):"]=ad
        output["van der Waals Forces (eV/Ang):"]= None    # [extra_prop] needs fill -> to_hdf5 molecule['pbe0FOR']
        output["Total Forces (eV/Ang):"]=F              # [extra_prop] total forces from forces.txt -> to_hdf5 molecule['totFOR']
      #   print (output)
        with open(jfile,'w') as fd:
            #   print("dumping json file ..")
              json.dump(output, fd, indent=4)
    
    # If anything fails, it returns an array of zeros
    except  Exception as e:
        print(f"DFTB calculation failed: {e}")
  
  
def get_atom_charges(ifile_path='detailed.out'):
    # Reads the detailed.out file produced by DFTB+
    ifile = open(ifile_path, 'r')
    lines = ifile.readlines()
    ifile.close()

    active = False
    c6p = []
    for line in lines:
        if not active:
            # Scans through detailed.out until it finds the block starting with "Atom Charge"
            if "Atom           Charge" in line:
                active = True
                nl = 0
        else:
            nl += 1
            c6pr = line.split()
            if len(c6pr) == 2:
                # Collects the second column = atomic charge
                c6p.append(float(c6pr[1]))
            else:
                # Stops when it encounters a line that doesn’t have exactly 2 entrie
                if nl > 3:
                    active = False
    # returns a Python list of per-atom charges
    return c6p


def get_band_energies(iorbs=4, ifile_path='band.out'):
    # Reads the band.out file produced by DFTB+
    # This file contains eigenvalues of the molecular orbitals
    ifile = open(ifile_path, 'r')
    lines = ifile.readlines()
    ifile.close()

    # c6p = list of [energy, occupation] pairs
    c6p = []
    active = False
    for line in lines:
        if not active:
            # Looks for the first k-point, spin channel section in band.out
            if "KPT" in line:
                # Once inside that section (active=True), it parses the eigenvalues
                active = True
                nl = 0
        else:
            nl += 1
            tokens = line.split()
            # Each valid line has 3 fields: index, eigenvalue, occupation
            if len(tokens) == 3:
                c6p.append([float(tokens[1]), float(tokens[2])])
            else:
                # Stops when the section ends
                if nl>3:
                    active = False
    c6p = np.array(c6p)
    
    # Identifies the HOMO–LUMO gap
    # HOMO = last occupied orbital (occupation ≠ 0)
    # LUMO = first unoccupied orbital (occupation = 0)
    HOMOs, LUMOs = [], []
    if len(c6p) >= 8:
        for i in range(len(c6p)):
            if c6p[i][1] == 0.0:
                # Collects <iorbs> HOMOs and <iorbs> LUMOs around the gap
                for j in range(iorbs):
                    HOMOs.append(c6p[i - j - 1][0])
                    LUMOs.append(c6p[i + j][0])
                break
        # 8 is a flag saying “I found a full set”
        return np.concatenate(([8], np.sort(HOMOs+LUMOs)), axis=None)
    else:
        # If too few orbitals (<8), it just returns whatever was found, preceded by the count
        return np.concatenate(([len(c6p)], np.sort(c6p[:,0])), axis=None)
