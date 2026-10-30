from __future__ import annotations

import math
from functools import reduce

# Property name -> key path inside the QM JSON written by the parsers.
PROP_JSON_KEYS = {
    "eMBD": ("SUM", "MBD"),
    "HLgap": ("HOMO-LUMO gap",),
    "DIP": ("Absolute Total Dipole (eAng):",),
    "eH": ("HOMO energy",),
    "eL": ("LUMO energy",),
}

# JSON entries that to_hdf5 reads unconditionally for the main dataset.
REQUIRED_JSON_KEYS = (
    ("Total energy (eV):",),
    ("SUM", "MBD"),
    ("Absolute Total Dipole (eAng):",),
    ("HOMO-LUMO gap",),
    ("HOMO energy",),
    ("LUMO energy",),
)

# Atomic reference energies used by to_hdf5; also defines the supported elements.
EPBE0_ATOM = {
    1: -13.64136507, 6: -1027.59344601, 7: -1484.25358313, 8: -2039.74560236,
    9: -3139.751675679, 15: -9283.01120606, 16: -10828.726222083, 17: -12516.462339357,
}

SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17}
SUPPORTED_SYMBOLS = frozenset(SYMBOL_TO_Z)

NUM_ATOMS_PROP = "atNUM"


def get_json_value(data, keys):
    """Walk a nested JSON dict along keys; returns None if any level is missing."""
    keys = keys if isinstance(keys, (list, tuple)) else (keys,)
    return reduce(lambda d, k: d.get(k) if isinstance(d, dict) else None, keys, data)


def is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
