import os
from os.path import join

import experiments.data.utils as dataset_utils
import numpy as np
import torch
from experiments.data.abstract_dataset import AbstractDataModule
from experiments.data.distributions import _labels
from experiments.data.metrics import compute_all_statistics
from experiments.data.utils import (
    load_pickle,
    save_pickle,
    train_subset,
)
from rdkit import RDLogger
from torch.utils.data import Subset
from torch_geometric.data import DataLoader, InMemoryDataset
from tqdm import tqdm

from src.eqdiff.concat import Concatenator, weights

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
#    "eC",
#    "eEE",
    "eH",
    "eKIN",
    # "eKSE",
    "eL",
#    "eNE",
#    "eNN",
    "eMBD",
#    "eTS",
#    "eX",
    "eXC",
    "eXX",
#    "mPOL",
#    "mC6",
]

atomic_energies_dict = {
    1: -13.6414041617373,
    6: -1027.60791501338,
    7: -1484.27481908749,
    8: -2039.75030551381,
    9: -2710.54734320680,
    15: -9283.01120605055,
    16: -10828.7228945312,
    17: -12516.4600457922,
}

atomic_numbers = [1, 6, 7, 8, 9, 15, 16, 17]
convert_z_to_x = {k: i for i, k in enumerate(atomic_numbers)}


class AQMQM7XDataset(InMemoryDataset):
    def __init__(
        self, split, root, remove_h, transform=None, pre_transform=None, pre_filter=None
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.remove_h = remove_h

        self.atom_encoder = full_atom_encoder
        if remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }

        self.label2idx = {k: i for i, k in enumerate(mol_properties)}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.post_processing()

    def post_processing(self):
        """load statistics and smiles"""
        self.statistics = dataset_utils.Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
            valencies=load_pickle(self.processed_paths[5]),
            bond_lengths=load_pickle(self.processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[7])),
            dihedrals=torch.from_numpy(np.load(self.processed_paths[8])).float(),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[10])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[11])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[12])

    @property
    def raw_file_names(self):
        if self.split == "train":
            return ["train_data.pickle"]
        elif self.split == "val":
            return ["val_data.pickle"]
        else:
            return ["test_data.pickle"]

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        confs = "all_confs"
        if self.split == "train":
            return [
                f"train_{h}_{confs}.pt",
                f"train_n_{h}_{confs}.pickle",
                f"train_atom_types_{h}_{confs}.npy",
                f"train_bond_types_{h}_{confs}.npy",
                f"train_charges_{h}_{confs}.npy",
                f"train_valency_{h}_{confs}.pickle",
                f"train_bond_lengths_{h}_{confs}.pickle",
                f"train_angles_{h}_{confs}.npy",
                f"train_dihedrals_{h}_{confs}.npy",
                f"train_is_aromatic_{h}_{confs}.npy",
                f"train_is_in_ring_{h}_{confs}.npy",
                f"train_hybridization_{h}_{confs}.npy",
                f"train_smiles_{confs}.pickle",
            ]
        elif self.split == "val":
            return [
                f"val_{h}_{confs}.pt",
                f"val_n_{h}_{confs}.pickle",
                f"val_atom_types_{h}_{confs}.npy",
                f"val_bond_types_{h}_{confs}.npy",
                f"val_charges_{h}_{confs}.npy",
                f"val_valency_{h}_{confs}.pickle",
                f"val_bond_lengths_{h}_{confs}.pickle",
                f"val_angles_{h}_{confs}.npy",
                f"val_dihedrals_{h}_{confs}.npy",
                f"val_is_aromatic_{h}_{confs}.npy",
                f"val_is_in_ring_{h}_{confs}.npy",
                f"val_hybridization_{h}_{confs}.npy",
                f"val_smiles_{confs}.pickle",
            ]
        else:
            return [
                f"test_{h}_{confs}.pt",
                f"test_n_{h}_{confs}.pickle",
                f"test_atom_types_{h}_{confs}.npy",
                f"test_bond_types_{h}_{confs}.npy",
                f"test_charges_{h}_{confs}.npy",
                f"test_valency_{h}_{confs}.pickle",
                f"test_bond_lengths_{h}_{confs}.pickle",
                f"test_angles_{h}_{confs}.npy",
                f"test_dihedrals_{h}_{confs}.npy",
                f"test_is_aromatic_{h}_{confs}.npy",
                f"test_is_in_ring_{h}_{confs}.npy",
                f"test_hybridization_{h}_{confs}.npy",
                f"test_smiles_{confs}.pickle",
            ]

    def download(self):
        raise ValueError(
            "Download and preprocessing is manual. If the data is already downloaded, "
            f"check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}"
        )

    def process(self):
        RDLogger.DisableLog("rdApp.*")
        all_data = load_pickle(self.raw_paths[0])

        data_list = []
        all_smiles = []
        for i, data in enumerate(tqdm(all_data)):
            data_list.append(data)

            all_smiles.append(data.smiles)

        torch.save(self.collate(data_list), self.processed_paths[0])

        statistics = compute_all_statistics(
            data_list,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
            additional_feats=True,
            include_force_norms=False,
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        np.save(self.processed_paths[8], statistics.dihedrals)
        np.save(self.processed_paths[9], statistics.is_aromatic)
        np.save(self.processed_paths[10], statistics.is_in_ring)
        np.save(self.processed_paths[11], statistics.hybridization)
        save_pickle(set(all_smiles), self.processed_paths[12])


class AQMQM7XDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.save_hyperparameters(cfg)
        self.datadir = cfg.dataset_root
        root_path = cfg.dataset_root
        self.pin_memory = True
        self.cfg = cfg

        self.label2idx = {k: i for i, k in enumerate(mol_properties)}

        self.concatenator = None
        if getattr(cfg, "concat_datasets", False):
            self.concatenator = Concatenator(
                AQMQM7XDataset,
                root_path,
                {"remove_h": cfg.remove_hs},
            )
            # names of the dataset_<n> folders, in the same order as the
            # children of the ConcatDatasets below
            self.dataset_names = [
                os.path.basename(r) for r in self.concatenator.dataset_roots()
            ]
            train_dataset = self.concatenator.concatenate(
                self.concatenator.datasets("train")
            )
            val_dataset = self.concatenator.concatenate(
                self.concatenator.datasets("val")
            )
            test_dataset = self.concatenator.concatenate(
                self.concatenator.datasets("test")
            )
            train_dataset.smiles = [
                smile
                for dataset in train_dataset.datasets
                for smile in dataset.smiles
            ]
        else:
            train_dataset = AQMQM7XDataset(
                split="train", root=root_path, remove_h=cfg.remove_hs
            )
            val_dataset = AQMQM7XDataset(
                split="val", root=root_path, remove_h=cfg.remove_hs
            )
            test_dataset = AQMQM7XDataset(
                split="test", root=root_path, remove_h=cfg.remove_hs
            )
        self.remove_h = cfg.remove_hs
        self.alpha = getattr(cfg, "alpha", 0.2)
        self.beta = getattr(cfg, "beta", 0.2)
        self.min_original_share = getattr(cfg, "min_original_share", 0.0) or 0.0
        # With concatenation, merge the statistics of all dataset_<n> folders
        # instead of only using dataset_0. num_nodes histograms are summed, so
        # new/larger molecule sizes stay in the support. The train marginals
        # (atom/bond/charge types = diffusion prior) are weighted with the same
        # weights as the sampler, so the prior matches what the model is
        # trained on; val/test are weighted by dataset size.
        if self.concatenator:
            sampler_weights = weights(
                train_dataset.datasets, self.alpha, self.beta, self.min_original_share
            )
            self.statistics = {
                "train": Concatenator.merge_statistics(
                    train_dataset.datasets, sampler_weights
                ),
                "val": Concatenator.merge_statistics(val_dataset.datasets),
                "test": Concatenator.merge_statistics(test_dataset.datasets),
            }
        else:
            self.statistics = {
                "train": train_dataset.statistics,
                "val": val_dataset.statistics,
                "test": test_dataset.statistics,
            }
        if cfg.select_train_subset:
            self.idx_train = train_subset(
                dset_len=len(train_dataset),
                train_size=cfg.train_size,
                seed=cfg.seed,
                filename=join(cfg.save_dir, "splits.npz"),
            )
            self.train_smiles = train_dataset.smiles
            train_dataset = Subset(train_dataset, self.idx_train)

        self.train_sampler = None
        self.samples_per_epoch = len(train_dataset)
        if self.concatenator:
            # samples_per_epoch only applies once there is more than one
            # dataset; the initial training (dataset_0 only) always covers the
            # full train split every epoch.
            if len(self.dataset_names) > 1 and getattr(cfg, "samples_per_epoch", None):
                self.samples_per_epoch = cfg.samples_per_epoch
            self.train_sampler = self.concatenator.sampler(
                train_dataset,
                alpha=self.alpha,
                beta=self.beta,
                num_samples=self.samples_per_epoch,
                min_first=self.min_original_share,
            )

        super().__init__(cfg, train_dataset, val_dataset, test_dataset)

    def train_dataloader(self, shuffle=True):
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle if self.train_sampler is None else False,
            sampler=self.train_sampler,
            persistent_workers=False,
        )
        return dataloader

    def sampling_summary(self):
        """Per-dataset sizes and sampling shares (written to sampling.json)."""
        if not self.concatenator:
            return {"concat_datasets": False, "samples_per_epoch": self.samples_per_epoch}
        train = self.train_dataset
        children = train.dataset.datasets if isinstance(train, Subset) else train.datasets
        shares = weights(children, self.alpha, self.beta, self.min_original_share)
        return {
            "concat_datasets": True,
            "alpha": self.alpha,
            "beta": self.beta,
            "min_original_share": self.min_original_share,
            "samples_per_epoch": self.samples_per_epoch,
            "datasets": [
                {
                    "name": name,
                    "share": share,
                    "n_train": len(child),
                    "n_val": len(val),
                    "n_test": len(test),
                    "draws_per_epoch": share * self.samples_per_epoch,
                    "draws_per_sample_per_epoch": share * self.samples_per_epoch / len(child),
                }
                for name, share, child, val, test in zip(
                    self.dataset_names,
                    shares,
                    children,
                    self.val_dataset.datasets,
                    self.test_dataset.datasets,
                )
            ],
        }

    @property
    def eval_names(self):
        """Logging suffix per val/test dataloader.

        Index 0 is always the full (concatenated) split and is logged under the
        plain stage name (e.g. "val/loss", which ModelCheckpoint monitors).
        With concatenation and more than one dataset, every dataset_<n> gets its
        own dataloader, logged as e.g. "val_dataset_1/loss".
        """
        names = [""]
        if self.concatenator and len(self.dataset_names) > 1:
            names += [f"_{name}" for name in self.dataset_names]
        return names

    def _eval_dataloaders(self, dataset, shuffle=False):
        datasets = [dataset]
        if len(self.eval_names) > 1:
            datasets += list(dataset.datasets)
        loaders = [
            DataLoader(
                dataset=d,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                pin_memory=self.pin_memory,
                shuffle=shuffle,
                persistent_workers=False,
            )
            for d in datasets
        ]
        return loaders[0] if len(loaders) == 1 else loaders

    def val_dataloader(self, shuffle=False):
        return self._eval_dataloaders(self.val_dataset, shuffle)

    def test_dataloader(self, shuffle=False):
        return self._eval_dataloaders(self.test_dataset, shuffle)

    def compute_mean_mad(self, properties_list):
        if self.hparams["dataset"] == "aqm_qm7x":
            dataloader = self.get_dataloader(self.train_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        elif (
            self.hparams["dataset"] == "aqm_half"
            or self.hparams["dataset"] == "aqm_2half"
        ):
            dataloader = self.get_dataloader(self.val_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        else:
            raise Exception("Wrong dataset name")

    def compute_mean_mad_from_dataloader(self, dataloader, properties_list):
        property_norms = {}
        for property_key in properties_list:
            idx = self.label2idx[property_key]

            # works for InMemoryDataset, ConcatDataset and Subset
            values_train = _labels(dataloader.dataset)[:, idx]
            mean = torch.mean(values_train)
            ma = torch.abs(values_train - mean)
            mad = torch.mean(ma)
            property_norms[property_key] = {}
            property_norms[property_key]["mean"] = mean
            property_norms[property_key]["mad"] = mad
            del values_train
        return property_norms

    def get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.cfg.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.cfg.inference_batch_size
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

        return dl
