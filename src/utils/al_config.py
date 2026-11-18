"""Loading and validation of the optional active-learning (`al:`) configuration.

Defaults live in src/configs/al.yaml; the experiment config only needs the keys it changes.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

NUM_ATOMS_PROP = "atNUM"

# Property name -> key path in the prop_calc JSON files (same paths as MolecularFilter.accuracy_filter).
PROPERTY_JSON_KEYS = {
    "eMBD": ("SUM", "MBD"),
    "HLgap": ("HOMO-LUMO gap",),
    "DIP": ("Absolute Total Dipole (eAng):",),
    "eH": ("HOMO energy",),
    "eL": ("LUMO energy",),
}
# Keys run_to_hdf5 reads unconditionally; molecules whose JSON lacks them are kept out of the HDF5.
REQUIRED_JSON_KEYS = (
    ("Total energy (eV):",),
    ("SUM", "MBD"),
    ("Absolute Total Dipole (eAng):",),
    ("HOMO-LUMO gap",),
    ("HOMO energy",),
    ("LUMO energy",),
)

STRATEGIES = ("trajectory", "kde_map", "frontier")
AL_ONLY_BLOCKS = ("plan", "dedup", "evaluate")

_DEFAULTS_FILE = Path(__file__).resolve().parents[1] / "configs" / "al.yaml"
DEFAULTS: dict[str, Any] = yaml.safe_load(_DEFAULTS_FILE.read_text())


def _merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def kde_counts(al: dict) -> tuple:
    """(candidates sampled, additional targets selected, anchor included) for kde_map.

    Anchor mode: al.anchor_n_candidates candidates in the box, al.anchor_n_targets selected from them, plus the
    trajectory row itself if al.include_anchor. Corridor mode: density.n_candidates and targets_per_iter.
    """
    if al.get("kde_region", "corridor") == "anchor":
        n_cand = al.get("anchor_n_candidates")
        n_cand = al["density"]["n_candidates"] if n_cand is None else n_cand
        return int(n_cand), int(al["anchor_n_targets"]), bool(al.get("include_anchor", True))
    return int(al["density"]["n_candidates"]), int(al["targets_per_iter"]), False


def get_al_cfg(cfg) -> dict:
    """`al:` section of the experiment config merged over the defaults (works when the section is absent)."""
    from omegaconf import OmegaConf

    raw = cfg.get("al", None)
    user = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    return _merge(DEFAULTS, user)


def resolve_tolerances(al: dict, prop_names: list, fallback: float) -> dict:
    """Per-property {relative, absolute} tolerances; a missing relative value falls back to `fallback`."""
    raw = al.get("tolerances", {}) or {}
    default = raw.get("default", {}) or {}
    resolved = {}
    for prop in prop_names:
        if prop == NUM_ATOMS_PROP:
            continue
        spec = raw.get(prop, default)
        if isinstance(spec, (int, float)):
            spec = {"relative": spec}
        relative = spec.get("relative", default.get("relative"))
        relative = fallback if relative is None else float(relative)
        absolute = float(spec.get("absolute", default.get("absolute", 0.0)) or 0.0)
        if relative < 0 or absolute < 0:
            raise ValueError(f"al.tolerances for {prop} must be non-negative")
        resolved[prop] = {"relative": relative, "absolute": absolute}
    return resolved


def tolerance_delta(target, prop_names: list, tolerances: dict) -> np.ndarray:
    """Allowed absolute error per property: max(relative * |target|, absolute)."""
    return np.array([max(tolerances[p]["relative"] * abs(float(v)), tolerances[p]["absolute"])
                     for p, v in zip(prop_names, np.asarray(target, dtype=float))])


def validate_al_cfg(cfg, al: dict, traj) -> None:
    """Fail early on settings the planner cannot run with."""
    if not al["enabled"]:
        return
    if al["query_strategy"] not in STRATEGIES:
        raise ValueError(f"al.query_strategy must be one of {STRATEGIES}, got {al['query_strategy']!r}")
    if not traj.has_properties:
        raise ValueError("Active learning needs property targets in the trajectory (query.props).")
    region = al.get("kde_region", "corridor")
    if region not in ("corridor", "anchor"):
        raise ValueError(f"al.kde_region must be 'corridor' or 'anchor', got {region!r}")
    if al["query_strategy"] == "kde_map" and region == "corridor" and len(traj) < 2:
        raise ValueError("al.query_strategy=kde_map needs at least 2 trajectory rows to define a corridor.")
    if region == "anchor" and not float(al["anchor_box"]) > 0:
        raise ValueError("al.anchor_box must be > 0")
    band = al["density"].get("density_percentiles")
    if band is not None:
        if len(band) != 2 or not 0.0 <= float(band[0]) < float(band[1]) <= 100.0:
            raise ValueError(f"al.density.density_percentiles must be [low, high] with 0 <= low < high <= 100, got {band}")
    n_cand, n_sel, with_anchor = kde_counts(al)
    sel_key = "anchor_n_targets" if region == "anchor" else "targets_per_iter"
    if n_sel < (0 if with_anchor else 1):
        raise ValueError(f"al.{sel_key} must be >= {0 if with_anchor else 1}")
    fraction = al.get("anchor_extra_fraction")
    fixed = region == "anchor" and with_anchor and fraction is not None
    if not fixed and n_sel + with_anchor > int(cfg.EQdiffParam.num_samples):
        raise ValueError(f"al.{sel_key}{' + the anchor' if with_anchor else ''} cannot exceed EQdiffParam.num_samples")
    if region == "anchor" and fraction is not None:
        total = int(cfg.EQdiffParam.num_samples)
        if not with_anchor:
            raise ValueError("al.anchor_extra_fraction needs al.include_anchor: true (num_samples goes to the anchor)")
        if not float(fraction) > 0.0:
            raise ValueError(f"al.anchor_extra_fraction must be > 0, got {fraction}")
        if round(float(fraction) * total) < 1:
            raise ValueError(f"al.anchor_extra_fraction={fraction} x num_samples={total} rounds to 0 molecules per "
                             f"additional target; raise the fraction or num_samples")
    if al.get("anchor_score", "kde") not in ("kde", "relative_pct"):
        raise ValueError(f"al.anchor_score must be 'kde' or 'relative_pct', got {al.get('anchor_score')!r}")
    if region == "anchor" and al.get("anchor_score") == "relative_pct":
        if NUM_ATOMS_PROP not in traj.plan_names:
            raise ValueError("al.anchor_score=relative_pct needs an atNUM column in the trajectory (and query.fix_num_nodes)")
        if float(al.get("anchor_min_datnum", 1)) < 0:
            raise ValueError("al.anchor_min_datnum must be >= 0")
    dedup = al.get("dedup", {}) or {}
    if dedup.get("enabled") and dedup.get("scope", "all") not in ("all", "target"):
        raise ValueError(f"al.dedup.scope must be 'all' or 'target', got {dedup.get('scope')!r}")
    if n_cand < n_sel:
        raise ValueError(f"The number of candidates ({n_cand}) must be >= al.{sel_key} ({n_sel})")
    unsupported = sorted(p for p in traj.prop_names if p not in PROPERTY_JSON_KEYS)
    if unsupported:
        raise ValueError(f"Properties {unsupported} cannot be read from the QM JSON; supported: {sorted(PROPERTY_JSON_KEYS)}")
    if not cfg.hdf5.appending_flag:
        raise ValueError("Active learning needs hdf5.appending_flag: True so that labels accumulate across iterations.")


def resolve_blocks(cfg, al: dict) -> list:
    """Pipeline blocks for the current mode.

    With al.enabled, `plan` is placed right before `generate` (after `train`) and `dedup` after `generate` (if
    al.dedup.enabled). Without it, the AL blocks are dropped. forward.py runs `plan` at that position, so an
    explicit `plan` elsewhere in pipeline.blocks is moved there too. `evaluate` is optional: it runs (after
    `hdf5`) only if it is listed in pipeline.blocks. `prop_filter` also runs only if listed; with al.enabled it
    compares each molecule with the target it was generated for (not the anchor row) and drops the misses.
    """
    blocks = list(cfg.pipeline.blocks)
    if not al["enabled"]:
        return [b for b in blocks if b not in AL_ONLY_BLOCKS]
    blocks = [b for b in blocks if b != "plan"]
    if "generate" in blocks:
        at = blocks.index("generate")
    elif "train" in blocks:
        at = blocks.index("train") + 1
    else:
        at = 0
    blocks.insert(at, "plan")
    if al.get("dedup", {}).get("enabled") and "dedup" not in blocks:
        after = next((blocks.index(b) + 1 for b in ("generate",) if b in blocks), None)
        before = next((blocks.index(b) for b in ("score_filter", "relax_geo", "prop_calc") if b in blocks), None)
        blocks.insert(after if after is not None else (before if before is not None else len(blocks)), "dedup")
    elif not al.get("dedup", {}).get("enabled"):
        blocks = [b for b in blocks if b != "dedup"]
    return blocks


def resolve_start(cfg, al: dict, blocks: list) -> str:
    """pipeline.start_at as given: `plan` sits after `train`, so any earlier start already includes it.

    start_at: plan re-plans (if targets.json was deleted; otherwise it is reused) without retraining;
    start_at: generate reuses the existing targets.json.
    """
    return cfg.pipeline.start_at


def al_paths(cfg) -> dict:
    """Files kept next to the experiment's configs/ folder."""
    exp_dir = Path(str(cfg.exp_config_path)).parent.parent
    return {
        "state_file": exp_dir / "al_state.json",
        "metrics_file": exp_dir / "al_metrics.csv",
    }
