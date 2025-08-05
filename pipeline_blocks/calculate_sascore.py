import os
import sys
from os import path
from os import environ, listdir, mkdir, makedirs, chdir, system, remove
import argparse
from pathlib import Path
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import RDConfig
try:
    # Construct the path to the SA_Score directory within RDKit's Contrib
    sa_score_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
    if sa_score_path not in sys.path:
        sys.path.append(sa_score_path)
    import sascorer
except ImportError as e:
    print(f"Error importing sascorer: {e}")
    print("Ensure that RDKit and its SA_Score module are properly installed.")
    sys.exit(1)


# def xyz_to_mol(xyz_file):
#     """
#     Convert an XYZ file to a MOL file using OpenBabel.

#     This function reads an XYZ file, converts it to a molecule object using OpenBabel,
#     and writes the molecule to a MOL file. If the conversion fails, it prints an error message.

#     Parameters
#     ----------
#     xyz_file : str
#         Path to the XYZ file.

#     Returns
#     -------
#     rdkit_mol : rdkit.Chem.rdchem.Mol
#         RDKit molecule object.
    
#     Raises
#     ------
#     Exception
#         If there is an error during the conversion process.
#     """
#     mol = next(pybel.readfile("xyz", str(xyz_file)))
#     mol.addh()
#     #mol.make3D()
#     #mol.localopt()
#     mol.OBMol.ConnectTheDots()
#     mol.OBMol.PerceiveBondOrders()
#     rdkit_mol = Chem.MolFromMolBlock(mol.write("mol"))

#     if rdkit_mol is None:
#         raise ValueError(f"Failed to convert {xyz_file} to RDKit molecule.")
#     return rdkit_mol



def xyz_to_smiles(xyz_file):
    """
    Convert an XYZ file to a SMILES string.

    This function reads an XYZ file, converts it to a molecule object using OpenBabel,
    and generates a SMILES string representation of the molecule. If the conversion fails,
    it prints an error message and returns None.

    Parameters
    ----------
    xyz_file : str
        Path to the XYZ file.

    Returns
    -------
    str or None
        SMILES string representation of the molecule, or None if the conversion fails.

    Raises
    ------
     Exception
        If there is an error during the conversion process.
    """
    try:
        with xyz_file.open('r') as f:
            xyz_content = f.read()
        mol_pybel = pybel.readstring("xyz", xyz_content)
        mol_pybel.OBMol.ConnectTheDots()
        mol_pybel.OBMol.PerceiveBondOrders()
        mol_pybel.addh() #optional
        smiles = mol_pybel.write("smi").strip()
        return smiles
    except Exception as e:
        print(f"Error converting XYZ to SMILES: {e}")
        return None
      

def calculate_sascore(smiles):
    """
    Calculate the SAScore for a given SMILES string.
    This function takes a SMILES string, converts it to an RDKit molecule object,
    and calculates the SAScore using the sascorer module. If the SMILES string is invalid
    or cannot be processed, it raises a ValueError and returns None.
    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    Returns
    -------
    float or None
        SAScore value, or None if the calculation fails.
    Raises
    ------
    ValueError
        If the SMILES string is invalid or cannot be processed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as e:
        return None, "tier_0"  # Invalid structure

    sascore = sascorer.calculateScore(mol)
    
    if sascore > 7:
        tier = "tier_1"
    elif sascore > 5:
        tier = "tier_2"
    elif sascore > 3:
        tier = "tier_3"
    else:
        tier = "tier_4"
    return sascore, tier



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate SAScore for molecules in XYZ format.")
    parser.add_argument("--input", type=str, help="Path to the input XYZ file")
    parser.add_argument("--output", type=str, help="Output folder for the SAScore results")
    args = parser.parse_args()

    try:
        ifile = Path(args.input)
        opath = Path(args.output)
    except:
        print("Usage:", sys.argv[0], "infile")
        sys.exit(1)
    if ifile.suffix == '.xyz':
        fname = ifile.name
        # mol = xyz_to_mol(ifile)
        # if mol:
        #     sascore, tier = calculate_sascore(mol)
        smiles = xyz_to_smiles(ifile)
        if smiles:
            sascore, tier  = calculate_sascore(smiles)
            with (opath / 'sascore.txt').open('a') as f:
                    f.write(f"{fname}\t{sascore}\t{tier}\n")
        else:
            print(f"Failed to convert {ifile} to SMILES.")
    else:
        print("Input file must be in XYZ format.")
        sys.exit(1)
                
