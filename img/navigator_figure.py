#!/usr/bin/env python3
"""Draw the Navigator example: a trajectory on the HLgap-eMBD map, coloured by atom count.

    python img/navigator_figure.py      # writes img/navigator-trajectory.png and img/navigator-trajectory-dark.png

The waypoints are read from Navigator/trajectory_HLgap_eMBD_atNUM.csv. The background is a synthetic
QM7-X-like distribution (illustrative); colours follow img/qalpa_figure.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "Navigator" / "trajectory_HLgap_eMBD_atNUM.csv"

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", ink2="#52514e", grid="#e1e0d9", al="#eb6834",
                  ramp=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", grid="#2c2c2a", al="#e0622f",
                 ramp=["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]),
}


def synthetic(seed=4, n=60000):
    rng = np.random.default_rng(seed)
    at = rng.gamma(4.0, 9.0, n) + 5                                   # atom count
    gap = 7.5 - 0.035 * at + rng.normal(0, 0.9, n) + rng.gamma(1.2, 0.35, n)
    embd = -0.035 * at + rng.normal(0, 0.18, n) * (1 + at / 60)
    keep = (gap > 1.5) & (gap < 12) & (embd > -3.6) & (embd < 0.1) & (at < 95)
    return gap[keep], embd[keep], at[keep]


def draw(theme):
    th = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    gap, embd, at = synthetic()
    way = np.genfromtxt(CSV, delimiter=",", names=True)
    cmap = LinearSegmentedColormap.from_list("blue", th["ramp"])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=th["bg"])
    ax.set_facecolor(th["bg"])
    hb = ax.hexbin(gap, embd, C=at, reduce_C_function=np.mean, gridsize=55, mincnt=3, cmap=cmap,
                   linewidths=0.2, edgecolors=th["bg"])
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("mean atom count (atNUM)", color=th["ink2"], fontsize=12)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=th["ink2"], labelsize=10.5)

    x, y = way["HLgap"], way["eMBD"]
    ax.plot(x, y, color=th["al"], lw=2.4, zorder=4, solid_capstyle="round")
    ax.scatter(x, y, s=95, c=th["al"], edgecolors=th["bg"], linewidths=2, zorder=5)
    n = len(x)
    offsets = [(12, -4, "left", "top"), (0, 14, "center", "bottom"), (-4, -14, "center", "top")]
    offsets += [(12, 0, "left", "center")] * max(n - 3, 0)
    for i, (xi, yi, ni) in enumerate(zip(x, y, way["atNUM"])):
        tag = "START · " if i == 0 else ("END · " if i == n - 1 else "")
        dx, dy, ha, va = offsets[i]
        ax.annotate(f"{tag}$q={i}$, {int(ni)} atoms", (xi, yi), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=10.5, color=th["ink"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc=th["bg"], ec="none", alpha=0.9))

    ax.set_xlabel("HOMO–LUMO gap, HLgap (eV)", color=th["ink2"], fontsize=12)
    ax.set_ylabel("MBD energy, eMBD (eV)", color=th["ink2"], fontsize=12)
    ax.tick_params(colors=th["ink2"], labelsize=10.5)
    for s in ax.spines.values():
        s.set_color(th["grid"])
    ax.grid(color=th["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Trajectory designed in the Navigator", color=th["ink"], fontsize=14, fontweight="bold",
                 loc="left")
    name = "navigator-trajectory" + ("" if theme == "light" else "-dark")
    fig.savefig(HERE / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=th["bg"])
    plt.close(fig)


if __name__ == "__main__":
    for t in THEMES:
        draw(t)
