import argparse
import os
import pickle
import random
import tempfile
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(parent_dir)
print(parent_dir)

import experiments.data.utils as dataset_utils
import h5py
import numpy as np
import torch
from experiments.data.utils import (
    get_rdkit_mol,
    write_xyz_file,
)
from rdkit import Chem
from tqdm import tqdm

full_atom_encoder = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
}
mol_properties = [
    "DIP",
    "HLgap",
    "eAT",
    # "eC",
    # "eEE",
    "eH",
    "eKIN",
 #   "eKSE",
    "eL",
    # "eNE",
    # "eNN",group_name
    "eMBD",
##    "eTS",
    # "eX",
    "eXC",
    "eXX",
    # "mPOL", 
]

def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--file-path', default=None, type=str, help='Data file path')
    parser.add_argument('--out-path', default=None, type=str, help='Processed data file path')
    parser.add_argument('--seed', default=42, type=int, help='Seed for the train/val/test split')
    parser.add_argument('--train-frac', default=0.8, type=float)
    parser.add_argument('--val-frac', default=0.1, type=float)
    args = parser.parse_args()
    return args


def read_properties(group, missing):
    """Read all `mol_properties`; absent ones become NaN.

    The DFTB+/EquiDTB backends do not write eKIN, eXC and eXX, so molecules
    added by the active-learning loop lack them. They are filled with NaN so
    every sample has the same `y` layout; only the properties in
    properties_list are used for conditioning.
    """
    values = []
    for prop in mol_properties:
        if prop in group:
            values.append(torch.tensor(np.array(group[prop]), dtype=torch.float32).reshape(-1)[:1])
        else:
            missing[prop] = missing.get(prop, 0) + 1
            values.append(torch.full((1,), float("nan")))
    return torch.cat(values).unsqueeze(0)


def split_sizes(n, train_frac, val_frac):
    """Number of molecules for train/val/test; val and test get >= 1 when n >= 3."""
    if n < 3:
        return n, 0, 0
    n_val = max(1, int(round(val_frac * n)))
    n_test = max(1, n - int(round(train_frac * n)) - n_val)
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def split_data(file_path, out_path, seed=42, train_frac=0.8, val_frac=0.1):
    # Group conformers by molecule (top-level HDF5 key) so that all conformers
    # of one molecule end up in the same split (no train/val/test leakage).
    molecules = {}
    missing = {}
    with h5py.File(file_path, "r") as h5:
        for mol_name, mol in tqdm(h5.items(), desc="Molecules"):
            for group_name in mol:
                group = mol[group_name]

                z = torch.tensor(np.array(group["atNUM"]), dtype=torch.long)
                all_pos = torch.tensor(np.array(group["atXYZ"]), dtype=torch.float32)
                y = read_properties(group, missing)

                assert all_pos.shape[0] == z.shape[0]
                assert all_pos.shape[1] == 3
                assert y.shape == (1, len(mol_properties))

                with tempfile.NamedTemporaryFile() as tmp:
                    tmp_file = tmp.name
                    # Write xyz file
                    write_xyz_file(coords=all_pos, atom_types=z, filename=tmp_file)
                    rdkit_mol = get_rdkit_mol(tmp_file)
                    smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)

                data = dataset_utils.mol_to_torch_geometric(
                    rdkit_mol,
                    full_atom_encoder,
                    smiles,
                    remove_hydrogens=False,
                )
                data.y = y
                data.mol = rdkit_mol
                molecules.setdefault(mol_name, []).append(data)

    if missing:
        print(f"Properties missing in {file_path}, filled with NaN: {missing}")

    names = sorted(molecules)
    if not names:
        raise ValueError(f"No molecules found in {file_path}")
    random.Random(seed).shuffle(names)

    n_train, n_val, n_test = split_sizes(len(names), train_frac, val_frac)
    train_names = names[:n_train]
    val_names = names[n_train : n_train + n_val]
    test_names = names[n_train + n_val :]
    if not val_names:
        # Too few molecules for a real split: reuse train for val/test so the
        # datasets are not empty (their val/test scores are then not held-out).
        print(
            f"WARNING: only {len(names)} molecule(s) in {file_path}; "
            "val/test are copies of train."
        )
        val_names = test_names = train_names

    def collect(selected):
        return [data for name in selected for data in molecules[name]]

    train_data, val_data, test_data = collect(train_names), collect(val_names), collect(test_names)
    print(
        f"Split {len(names)} molecules -> train/val/test molecules "
        f"{len(train_names)}/{len(val_names)}/{len(test_names)}, conformers "
        f"{len(train_data)}/{len(val_data)}/{len(test_data)}"
    )

    os.makedirs(os.path.join(out_path, "raw"), exist_ok=True)

    with open(os.path.join(out_path, "raw/train_data.pickle"), "wb") as f:
        pickle.dump(train_data, f)

    with open(os.path.join(out_path, "raw/val_data.pickle"), "wb") as f:
        pickle.dump(val_data, f)

    with open(os.path.join(out_path, "raw/test_data.pickle"), "wb") as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    args = get_args()
    split_data(args.file_path, args.out_path, args.seed, args.train_frac, args.val_frac)
