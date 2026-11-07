"""Active-learning configuration loading, validation, and workflow helpers.

Static defaults are kept in ``al.yaml``; runtime
configuration behavior stays together here so the configuration flow is easy
to follow from one module.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import yaml

from src.components.dataset_labels import dataset_values, load_label_dataset

if TYPE_CHECKING:
    from omegaconf import DictConfig

_DEFAULTS_FILE = Path(__file__).with_name("al.yaml")
_loaded_defaults = yaml.safe_load(_DEFAULTS_FILE.read_text())
if not isinstance(_loaded_defaults, dict):
    raise ValueError(f"Active-learning defaults must be a mapping: {_DEFAULTS_FILE}")
DEFAULTS: dict[str, Any] = _loaded_defaults

LEGACY_BLOCKS = ["split", "train", "generate", "score_filter", "relax_geo", "prop_calc", "prop_filter", "hdf5", "compress"]
AL_BLOCKS = ["plan", "split", "train", "generate", "score_filter", "acquire", "relax_geo", "prop_calc", "hdf5", "evaluate", "compress"]
AL_ONLY_BLOCKS = {"plan", "acquire", "evaluate"}
STRATEGIES = ("trajectory", "kde_map", "frontier")
ACQ_MODES = ("score", "random")


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge an override into a copied configuration."""
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def get_al_cfg(cfg: "DictConfig") -> dict:
    """Resolve active-learning settings, including properties from EQdiff config."""
    from omegaconf import OmegaConf

    raw = cfg.get("al", None)
    user = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    al = _merge(DEFAULTS, user)
    if al["surrogate"].get("props") is None:
        ep = cfg.get("EQdiffParam", None)
        props = ep.get("properties_list", None) if ep is not None else None
        al["surrogate"]["props"] = list(props) if props is not None else []
    dataset_path = al.get("dataset_config")
    al["dataset"] = load_label_dataset(dataset_path)
    return al


def al_enabled(cfg: "DictConfig") -> bool:
    return bool(get_al_cfg(cfg)["enabled"])


def resolve_tolerances(al: dict, prop_names: list[str], fallback: float) -> dict:
    """Resolve per-property relative/absolute tolerances with a legacy fallback."""
    raw = al.get("tolerances", {}) or {}
    default = raw.get("default", {}) or {}
    resolved = {}
    for prop in prop_names:
        spec = raw.get(prop, default)
        if isinstance(spec, (int, float)):
            spec = {"relative": spec}
        relative = spec.get("relative", fallback)
        if relative is None:
            relative = fallback
        absolute = spec.get("absolute", 0.0)
        relative, absolute = float(relative), float(absolute)
        if relative < 0 or absolute < 0:
            raise ValueError(f"Tolerance for {prop} must be non-negative")
        resolved[prop] = {"relative": relative, "absolute": absolute}
    return resolved


def tolerance_delta(target: np.ndarray, prop_names: list[str], tolerances: dict) -> np.ndarray:
    """Return the allowed absolute error for each target property."""
    return np.array([
        max(tolerances[prop]["relative"] * abs(float(value)), tolerances[prop]["absolute"])
        for prop, value in zip(prop_names, np.asarray(target, dtype=float))
    ])


def validate_al_cfg(al: dict, num_samples: int, prop_names: list[str], dataset: dict | None = None) -> None:
    if not al["enabled"]:
        return
    values = dataset_values(dataset or al.get("dataset"))
    num_atoms_prop = values["num_atoms_property"]
    property_keys = values["property_json_keys"]
    if al["query_strategy"] not in STRATEGIES:
        raise ValueError(f"al.query_strategy must be one of {STRATEGIES}, got {al['query_strategy']!r}")
    if al["acquisition"]["mode"] not in ACQ_MODES:
        raise ValueError(f"al.acquisition.mode must be one of {ACQ_MODES}, got {al['acquisition']['mode']!r}")
    if int(al["targets_per_iter"]) < 1:
        raise ValueError("al.targets_per_iter must be >= 1")
    if int(al["oversample"]) < 1:
        raise ValueError("al.oversample must be >= 1")
    pool = int(num_samples) * int(al["oversample"])
    if int(al["budget"]) < 1 or int(al["budget"]) > pool:
        raise ValueError(f"al.budget ({al['budget']}) must be in [1, EQdiffParam.num_samples*al.oversample = {pool}]")
    if int(al["targets_per_iter"]) > pool:
        raise ValueError(f"al.targets_per_iter ({al['targets_per_iter']}) exceeds the candidate pool ({pool})")
    wanted = list(al["surrogate"]["props"]) + list(prop_names)
    unsupported = sorted({p for p in wanted if p != num_atoms_prop and p not in property_keys})
    if unsupported:
        raise ValueError(f"Properties {unsupported} cannot be read from the QM JSON; supported: {sorted(property_keys)}")
    surrogate_props = list(al["surrogate"]["props"])
    missing = [p for p in surrogate_props if p not in prop_names]
    if missing:
        raise ValueError(f"al.surrogate.props must be a subset of query.props; not found: {missing}")
    if not any(p != num_atoms_prop for p in surrogate_props):
        raise ValueError("al.surrogate.props must include at least one non-atNUM property")
    resolve_tolerances(al, prop_names, 0.1)


def resolve_blocks(cfg: "DictConfig") -> list[str]:
    """Return workflow blocks appropriate for the active-learning mode."""
    blocks = list(cfg.pipeline.blocks)
    enabled = al_enabled(cfg)
    if not enabled:
        if blocks == AL_BLOCKS:
            return list(LEGACY_BLOCKS)
        return [block for block in blocks if block not in AL_ONLY_BLOCKS]
    if blocks == LEGACY_BLOCKS:
        return list(AL_BLOCKS)
    return [block for block in blocks if block != "prop_filter"]


def resolve_start(cfg: "DictConfig", blocks: list[str]) -> str:
    """Ensure legacy ``split`` does not skip target planning when AL is enabled."""
    start = cfg.pipeline.start_at
    if al_enabled(cfg) and start == LEGACY_BLOCKS[0] and "plan" in blocks:
        return "plan"
    return start


def al_paths(cfg: "DictConfig") -> dict:
    """Return sidecar paths next to the experiment configuration."""
    exp_dir = Path(str(cfg.exp_config_path)).parent.parent
    return {
        "exp_dir": exp_dir,
        "state_file": exp_dir / "al_state.json",
        "validation_file": exp_dir / "validation_ids.json",
        "metrics_file": exp_dir / "metrics.csv",
        "cache_dir": exp_dir / "al_cache",
    }


def should_stop(cfg: "DictConfig") -> bool:
    loop = cfg.get("loop", None)
    return bool(loop.get("stop", False)) if loop is not None else False


def request_stop(cfg: "DictConfig") -> None:
    from omegaconf import open_dict

    with open_dict(cfg.loop):
        cfg.loop.stop = True
