"""Dataset label schemas shared by conversion, active learning, and training."""

from __future__ import annotations

import json
import math
from functools import reduce

from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

_DEFAULT_DATASET = Path(__file__).parent.parent / "configs" / "datasets" / "qm7.yaml"


def load_label_dataset(path=None) -> dict:
    """Load a dataset label schema, defaulting to the bundled QM7 schema."""
    config_path = Path(path) if path is not None else _DEFAULT_DATASET
    with config_path.open("r") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset configuration must be a mapping: {config_path}")
    return config


def dataset_values(dataset=None) -> dict:
    """Normalize schema values used by label extraction and validation."""
    config = dataset or load_label_dataset()
    return {
        "property_json_keys": {key: tuple(value) for key, value in config["property_json_keys"].items()},
        "required_json_keys": tuple(tuple(value) for value in config["required_json_keys"]),
        "atomic_reference_energies": {
            int(key): float(value) for key, value in config["atomic_reference_energies"].items()
        },
        "symbol_to_atomic_number": {
            str(key): int(value) for key, value in config["symbol_to_atomic_number"].items()
        },
        "num_atoms_property": str(config["num_atoms_property"]),
    }


NUM_ATOMS_PROP = load_label_dataset()["num_atoms_property"]


def get_json_value(data, keys):
    """Walk a nested JSON mapping along a key path."""
    keys = keys if isinstance(keys, (list, tuple)) else (keys,)
    return reduce(lambda current, key: current.get(key) if isinstance(current, dict) else None, keys, data)


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def label_names(dataset: dict | None = None) -> list[str]:
    """Return labels intended as model targets for a dataset."""
    config = dataset or load_label_dataset()
    return list(config.get("training_labels", config.get("hdf5_labels", {}).keys()))


def _label_value(spec: dict[str, Any], data: dict, atomic_numbers: np.ndarray,
                 dataset: dict) -> Any:
    if "path" in spec:
        return get_json_value(data, spec["path"])
    derived = spec.get("derived")
    if derived == "pbe0_energy":
        return np.asarray([float(get_json_value(data, ["Total energy (eV):"])) -
                           float(get_json_value(data, ["SUM", "MBD"]))])
    if derived == "atomization_energy":
        total = float(get_json_value(data, ["Total energy (eV):"])) - float(
            get_json_value(data, ["SUM", "MBD"])
        )
        references = dataset_values(dataset)["atomic_reference_energies"]
        return np.asarray([total - sum(references[int(z)] for z in atomic_numbers)])
    if derived == "difference":
        minuend = get_json_value(data, spec["minuend"])
        subtrahend = get_json_value(data, spec["subtrahend"])
        return np.asarray(minuend, dtype=float) - np.asarray(subtrahend, dtype=float)
    if derived == "ratio":
        numerator = np.asarray(get_json_value(data, spec["numerator"]), dtype=float)
        denominator = np.asarray(get_json_value(data, spec["denominator"]), dtype=float)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    raise ValueError(f"Unsupported dataset label definition: {spec}")


def extract_labels(data: dict, atomic_numbers, dataset: dict | None = None,
                   group: str = "hdf5_labels") -> dict[str, np.ndarray]:
    """Extract configured scalar/vector labels from one parsed QM result."""
    config = dataset or load_label_dataset()
    values = {}
    for name, spec in config.get(group, {}).items():
        try:
            raw = _label_value(spec, data, np.asarray(atomic_numbers), config)
        except (KeyError, TypeError, ValueError):
            if spec.get("optional", False):
                continue
            raise
        if raw is None or not np.all(np.isfinite(np.asarray(raw, dtype=float))):
            if spec.get("optional", False):
                continue
            raise ValueError(f"Missing or non-finite dataset label: {name}")
        values[name] = np.atleast_1d(np.asarray(raw, dtype=np.float64))
    return values


def read_xyz_symbols(path) -> list:
    """Read element symbols from an XYZ file."""
    with Path(path).open("r") as stream:
        n = int(stream.readline().split()[0])
        stream.readline()
        return [stream.readline().split()[0] for _ in range(n)]


def read_xyz(path) -> tuple:
    """Read element symbols and positions from an XYZ file."""
    with Path(path).open("r") as stream:
        n = int(stream.readline().split()[0])
        stream.readline()
        symbols, positions = [], []
        for _ in range(n):
            parts = stream.readline().split()
            symbols.append(parts[0])
            positions.append([float(value) for value in parts[1:4]])
    return symbols, positions


def read_json_props(json_path, prop_names, num_atoms: Optional[int] = None,
                    dataset: Optional[dict] = None) -> Optional[dict]:
    """Read configured target properties from a QM JSON result."""
    path = Path(json_path)
    if not path.exists():
        return None
    try:
        with path.open("r") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return None
    values = dataset_values(dataset)
    result = {}
    for name in prop_names:
        if name == values["num_atoms_property"]:
            if num_atoms is None:
                return None
            result[name] = float(num_atoms)
            continue
        value = get_json_value(data, values["property_json_keys"].get(name, ()))
        if not is_finite_number(value):
            return None
        result[name] = float(value)
    return result


def valid_qm_result(json_path, xyz_path, prop_names, dataset: Optional[dict] = None) -> tuple:
    """Validate a QM JSON result against the selected dataset schema."""
    path = Path(json_path)
    if not path.exists():
        return False, "missing QM json"
    try:
        symbols = read_xyz_symbols(xyz_path)
    except (OSError, ValueError, IndexError):
        return False, "unreadable xyz"
    values = dataset_values(dataset)
    supported = frozenset(values["symbol_to_atomic_number"])
    if not set(symbols) <= supported:
        return False, f"unsupported elements {sorted(set(symbols) - supported)}"
    try:
        with path.open("r") as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return False, "unreadable QM json"
    for keys in values["required_json_keys"]:
        if not is_finite_number(get_json_value(data, keys)):
            return False, f"missing or non-finite {'/'.join(keys)}"
    if read_json_props(path, prop_names, num_atoms=len(symbols), dataset=dataset) is None:
        return False, "missing target property"
    return True, ""
