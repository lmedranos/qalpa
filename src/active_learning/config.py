from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig

from src.active_learning.constants import NUM_ATOMS_PROP, PROP_JSON_KEYS

LEGACY_BLOCKS = ["split", "train", "generate", "score_filter", "relax_geo", "prop_calc", "prop_filter", "hdf5", "compress"]
AL_BLOCKS = ["plan", "split", "train", "generate", "score_filter", "acquire", "relax_geo", "prop_calc", "hdf5", "evaluate", "compress"]
AL_ONLY_BLOCKS = {"plan", "acquire", "evaluate"}

STRATEGIES = ("trajectory", "kde_map", "frontier")
ACQ_MODES = ("score", "random")

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "query_strategy": "kde_map",
    "targets_per_iter": 4,
    "oversample": 5,
    "budget": 200,
    "corridor_pad": 0.1,
    "corridor_lookahead": None,
    "advance_hit_rate": 0.3,
    "keep_off_target": True,
    "generation_budget_hours": 12,
    "min_neff": 5,
    "tolerances": {
        "default": {"relative": None, "absolute": 0.0},
    },
    "density": {
        "bandwidth": "scott",
        "n_candidates": 2000,
        "w_sparse": 0.5,
        "w_err": 0.5,
        "w_explore": 0.5,
        "min_reach": 0.05,
        "logp_floor": -20.0,
        "reweight": False,
    },
    "acquisition": {
        "mode": "score",
        "w_hit": 1.0,
        "w_unc": 0.5,
        "w_nov": 0.5,
        "w_div": 0.5,
        "w_cost": 0.1,
        "novelty_ref_max": 20000,
    },
    "surrogate": {
        "backend": "fingerprint_rf",
        "n_members": 5,
        "n_trees": 30,
        "min_train": 50,
        "max_train": 20000,
        "props": None,
    },
    "validation": {"size": 200},
    "early_stop": {"enabled": False, "patience": 3, "min_delta": 0.01},
}


def _merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def get_al_cfg(cfg: DictConfig) -> dict:
    """Resolved AL settings as a plain dict; older configs without `al:` give the disabled defaults."""
    from omegaconf import OmegaConf

    raw = cfg.get("al", None)
    user = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    al = _merge(DEFAULTS, user)
    if al["surrogate"].get("props") is None:
        ep = cfg.get("EQdiffParam", None)
        props = ep.get("properties_list", None) if ep is not None else None
        al["surrogate"]["props"] = list(props) if props is not None else []
    return al


def al_enabled(cfg: DictConfig) -> bool:
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


def validate_al_cfg(al: dict, num_samples: int, prop_names: list[str]) -> None:
    if not al["enabled"]:
        return
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
    unsupported = sorted({p for p in wanted if p != NUM_ATOMS_PROP and p not in PROP_JSON_KEYS})
    if unsupported:
        raise ValueError(f"Properties {unsupported} cannot be read from the QM JSON; supported: {sorted(PROP_JSON_KEYS)}")
    surrogate_props = list(al["surrogate"]["props"])
    missing = [p for p in surrogate_props if p not in prop_names]
    if missing:
        raise ValueError(f"al.surrogate.props must be a subset of query.props; not found: {missing}")
    if not any(p != NUM_ATOMS_PROP for p in surrogate_props):
        raise ValueError("al.surrogate.props must include at least one non-atNUM property")
    resolve_tolerances(al, prop_names, 0.1)


def resolve_blocks(cfg: DictConfig) -> list[str]:
    """Pipeline blocks for the active mode; the legacy default list is swapped for the AL list when enabled."""
    blocks = list(cfg.pipeline.blocks)
    if not al_enabled(cfg):
        if blocks == AL_BLOCKS:
            return list(LEGACY_BLOCKS)
        return [b for b in blocks if b not in AL_ONLY_BLOCKS]
    if blocks == LEGACY_BLOCKS:
        return list(AL_BLOCKS)
    return [b for b in blocks if b != "prop_filter"]


def al_paths(cfg: DictConfig) -> dict:
    """Sidecar locations for AL state; they live next to the experiment config, never in `cfg`."""
    exp_dir = Path(str(cfg.exp_config_path)).parent.parent
    return {
        "exp_dir": exp_dir,
        "state_file": exp_dir / "al_state.json",
        "validation_file": exp_dir / "validation_ids.json",
        "metrics_file": exp_dir / "metrics.csv",
        "cache_dir": exp_dir / "al_cache",
    }


def resolve_start(cfg: DictConfig, blocks: list[str]) -> str:
    """Resume point; the template's legacy default `split` must not skip `plan` when AL is enabled."""
    start = cfg.pipeline.start_at
    if al_enabled(cfg) and start == LEGACY_BLOCKS[0] and "plan" in blocks:
        return "plan"
    return start


def should_stop(cfg: DictConfig) -> bool:
    loop = cfg.get("loop", None)
    return bool(loop.get("stop", False)) if loop is not None else False


def request_stop(cfg: DictConfig) -> None:
    from omegaconf import open_dict

    with open_dict(cfg.loop):
        cfg.loop.stop = True
