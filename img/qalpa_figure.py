#!/usr/bin/env python3
"""Draw the QALPA overview figure.

    python img/qalpa_figure.py              # writes img/qalpa-workflow.png and img/qalpa-workflow-dark.png
    python img/qalpa_figure.py --svg        # also writes the SVG versions

The property-space panels use a synthetic dataset. The target selection in the zoom panel runs the rule of
src/pipeline_blocks/plan_targets.py: candidates in a box around the waypoint, a density band of the initial
data (5th to 15th percentile), the relative_pct score and farthest-point picks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import (Circle, ConnectionPatch, Ellipse, FancyArrowPatch,  # noqa: E402
                                FancyBboxPatch, Polygon, Rectangle)

W, H = 160.0, 90.0                     # canvas units; 1 unit = 0.1 inch
DPI = 200

THEMES = {
    "light": dict(
        bg="#ffffff", card="#ffffff", frame="#f6f7f8", ink="#1f2328", ink2="#52514e", muted="#8e8a80",
        edge="#b9b5ab", fwd="#2a78d6", fwd_tint="#f1f6fd", al="#eb6834", al_soft="#f6b08f",
        bwd="#1baf7a", bwd_tint="#eef9f4", bwd_soft="#bfe8d6", good="#0a8f0a", bad="#d03b3b",
        data_fill="#f4f3f0", contour="#ffffff", band="#f8c6aa",
        density=["#f4f3f0", "#ebe9e4", "#e0ddd5", "#d2cec4", "#c2bdb1", "#afa99b", "#999283", "#7f786a"],
        atom="#3c3c3a", atom2="#8a877f", bond="#6f6c66", replay=["#184f95", "#5598e7", "#9ec5f4"],
    ),
    "dark": dict(
        bg="#0d1117", card="#1a1a19", frame="#232322", ink="#f0efec", ink2="#c3c2b7", muted="#8e8a80",
        edge="#4a4944", fwd="#3987e5", fwd_tint="#141c27", al="#e0622f", al_soft="#8a4527",
        bwd="#199e70", bwd_tint="#122019", bwd_soft="#1f5a43", good="#2fbf2f", bad="#e66767",
        data_fill="#2a2a28", contour="#1a1a19", band="#7a3f22",
        density=["#262624", "#2e2e2b", "#383733", "#43423d", "#504e48", "#605d55", "#726f65", "#878377"],
        atom="#e6e4dd", atom2="#a8a59c", bond="#b3b0a6", replay=["#9ec5f4", "#5598e7", "#256abf"],
    ),
}

FS = dict(title=15.2, lane=13.5, body=13, small=12, tiny=11)


# ----------------------------------------------------------------------------------------------- data + KDE
def make_data(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def blob(n, mean, s_major, s_minor, deg):
        a = np.deg2rad(deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return (rng.normal(size=(n, 2)) * [s_major, s_minor]) @ rot.T + mean

    pts = np.vstack([blob(1500, (0.45, 0.47), 0.18, 0.065, 38),
                     blob(650, (0.29, 0.31), 0.07, 0.045, 25),
                     blob(220, (0.66, 0.70), 0.08, 0.035, 48)])
    return pts[(pts > 0.03).all(1) & (pts < 0.97).all(1)]


BW = 0.045                              # KDE bandwidth in map units


def kde(pts: np.ndarray, q: np.ndarray, h: float = BW, chunk: int = 1024) -> np.ndarray:
    q = np.atleast_2d(q)
    out = np.empty(len(q))
    for s in range(0, len(q), chunk):
        d2 = ((q[s:s + chunk, None, :] - pts[None, :, :]) ** 2).sum(-1)
        out[s:s + chunk] = np.exp(-0.5 * d2 / h ** 2).sum(1)
    return out / (len(pts) * 2.0 * np.pi * h * h)


def physical(z: np.ndarray) -> np.ndarray:
    """Map units -> property units (HOMO-LUMO gap and MBD energy in eV), for the relative-% score."""
    z = np.atleast_2d(z)
    return np.column_stack([2.0 + 8.0 * z[:, 0], -3.5 + 3.5 * z[:, 1]])


def plan(pts, anchor, half, band, rng, n_cand=160, n_targets=2):
    """Candidates in the anchor box, density band, relative_pct score, farthest-point picks (plan_targets.py)."""
    lo, hi = band
    cand = anchor + (rng.random((n_cand, 2)) * 2.0 - 1.0) * half
    a = physical(anchor)[0]
    rel = 100.0 * (physical(cand) - a) / np.abs(a)
    dc, da = kde(pts, cand), kde(pts, anchor)[0]
    score = np.abs(rel).sum(1) + 100.0 * np.abs(dc - da) / da
    eligible = (dc >= lo) & (dc <= hi)
    mind = np.linalg.norm(rel, axis=1)            # distance to the anchor (its relative difference is 0)
    avail, picked = eligible.copy(), []
    for k in range(n_targets):
        if not avail.any():
            break
        key = score if k == 0 else mind
        i = int(np.argmax(np.where(avail, key, -np.inf)))
        picked.append(i)
        avail[i] = False
        mind = np.minimum(mind, np.linalg.norm(rel - rel[i], axis=1))
    return cand, eligible, picked


# ------------------------------------------------------------------------------------------------ drawing kit
def card(ax, x, y, w, h, th, edge=None, lw=1.2, fc=None, r=1.4, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc or th["card"], ec=edge or th["edge"], lw=lw, zorder=z))


def step_title(ax, x, y, n, title, color, th, sub=None):
    ax.add_patch(Circle((x, y), 1.45, fc=color, ec="none", zorder=5))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=11.5, fontweight="bold", color="white", zorder=6)
    ax.text(x + 2.3, y, title, ha="left", va="center", fontsize=FS["title"], fontweight="bold", color=th["ink"],
            zorder=6)
    if sub:
        ax.text(x - 1.45, y - 3.6, sub, ha="left", va="center", fontsize=FS["small"], color=th["ink2"], zorder=6)


def arrow(ax, p, q, th, color=None, lw=1.6, ms=15, ls="-", z=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color or th["ink2"],
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))


def lane_label(ax, x, y, tag, text, color, th):
    ax.add_patch(FancyBboxPatch((x, y - 1.25), 5.8, 2.5, boxstyle="round,pad=0,rounding_size=1.25", fc=color,
                                ec="none", zorder=5))
    ax.text(x + 2.9, y - 0.05, tag, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white",
            zorder=6)
    ax.text(x + 7.2, y, text, ha="left", va="center", fontsize=FS["lane"], fontweight="bold", color=th["ink"],
            zorder=6)


def cylinder(ax, x, y, w, h, label, th, fs=13):
    e = h * 0.24
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=th["data_fill"], ec=th["ink2"], lw=1.5, zorder=3))
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=th["data_fill"], ec="none", zorder=4))
    for xx in (x, x + w):
        ax.plot([xx, xx], [y + e / 2, y + h - e / 2], color=th["ink2"], lw=1.5, zorder=5, solid_capstyle="butt")
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc=th["data_fill"], ec=th["ink2"], lw=1.5, zorder=5))
    ax.text(x + w / 2, y + (h - e) / 2 + 0.1, label, ha="center", va="center", fontsize=fs, fontweight="bold",
            color=th["ink"], zorder=6, linespacing=1.15)


def diamond(ax, cx, cy, hw, hh, text, th):
    ax.add_patch(Polygon([[cx - hw, cy], [cx, cy + hh], [cx + hw, cy], [cx, cy - hh]], closed=True,
                         fc=th["card"], ec=th["ink2"], lw=1.5, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=14, color=th["ink"], zorder=4)


def pill(ax, cx, cy, w, text, th, edge):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - 1.6), w, 3.2, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=th["card"], ec=edge, lw=1.1, zorder=4))
    ax.text(cx, cy - 0.05, text, ha="center", va="center", fontsize=FS["small"], color=th["ink"], zorder=5)


def molecule(ax, cx, cy, r, th, rot=0.0, jitter=0.0, alpha=1.0, rng=None, z=6):
    """Skeletal glyph: an aromatic six-ring with three substituent atoms and a short chain."""
    ang = np.deg2rad(np.arange(6) * 60.0 + 30.0 + rot)
    ring = np.column_stack([np.cos(ang), np.sin(ang)])
    subs = ring[[0, 2, 4]] * 1.9
    tail = subs[0] + (subs[0] - ring[0]) * 0.8
    atoms = np.vstack([ring, subs, tail[None]])
    bonds = [(i, (i + 1) % 6) for i in range(6)] + [(0, 6), (2, 7), (4, 8), (6, 9)]
    if jitter and rng is not None:
        atoms = atoms + rng.normal(scale=jitter, size=atoms.shape)
    p = atoms * r + [cx, cy]
    for i, j in bonds:
        ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color=th["bond"], lw=1.8, alpha=alpha, zorder=z,
                solid_capstyle="round")
    if not jitter:
        ax.add_patch(Circle((cx, cy), 0.55 * r, fc="none", ec=th["bond"], lw=1.3, alpha=alpha, zorder=z))
    ax.scatter(p[6:, 0], p[6:, 1], s=np.array([30.0, 30.0, 30.0, 24.0]) * r * r,
               c=[th["atom2"], th["atom"], th["atom2"], th["atom"]], edgecolors=th["card"], linewidths=0.8,
               alpha=alpha, zorder=z + 1)


def noise(ax, x, y, s, th, rng, n=26, z=6):
    xy = np.column_stack([x + 0.8 + rng.random(n) * (s - 1.6), y + 0.8 + rng.random(n) * (s - 1.6)])
    rgba = np.array([to_rgba(th["atom"])] * n)
    rgba[:, 3] = 0.3 + 0.6 * rng.random(n)
    ax.scatter(xy[:, 0], xy[:, 1], s=12 + 16 * rng.random(n), c=rgba, edgecolors="none", zorder=z)


def map_axes(fig, x, y, w, h, xlim=(0, 1), ylim=(0, 1)):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_xlim(*xlim)
    a.set_ylim(*ylim)
    a.axis("off")
    a.set_zorder(3)
    return a


def axis_arrows(ax, x, y, w, h, th):
    arrow(ax, (x, y), (x + w + 0.8, y), th, lw=1.2, ms=10)
    arrow(ax, (x, y), (x, y + h + 0.8), th, lw=1.2, ms=10)
    ax.text(x + w + 1.2, y, "$P_1$", ha="left", va="center", fontsize=FS["small"], color=th["ink2"], zorder=6)
    ax.text(x - 0.6, y + h, "$P_2$", ha="right", va="center", fontsize=FS["small"], color=th["ink2"], zorder=6)


def density(axm, gx, gy, g, levels, th):
    top = max(float(g.max()), float(levels[-1])) * 1.01        # keep the levels increasing in small windows
    axm.contourf(gx, gy, g, levels=list(levels) + [top], colors=th["density"][:len(levels)], zorder=1)
    axm.contour(gx, gy, g, levels=levels, colors=[th["contour"]], linewidths=0.5, zorder=2)


def trajectory(axm, way, cur, th, faint=False):
    a = 0.6 if faint else 1.0
    axm.plot(way[:cur + 1, 0], way[:cur + 1, 1], color=th["ink2"], lw=1.8, alpha=a, zorder=4,
             solid_capstyle="round")
    axm.plot(way[cur:, 0], way[cur:, 1], color=th["ink2"], lw=1.6, ls=(0, (2.5, 2.0)), alpha=a, zorder=4)
    axm.scatter(way[:cur, 0], way[:cur, 1], s=34, c=th["ink"], edgecolors=th["card"], linewidths=1.0, alpha=a,
                zorder=5)
    axm.scatter(way[cur + 1:, 0], way[cur + 1:, 1], s=34, c=th["card"], edgecolors=th["ink"], linewidths=1.3,
                alpha=a, zorder=5)
    axm.scatter([way[cur, 0]], [way[cur, 1]], marker="*", s=200, c=th["ink"], edgecolors=th["card"],
                linewidths=1.0, alpha=a, zorder=6)


# ------------------------------------------------------------------------------------------------------ figure
def draw(theme: str, out: Path, svg: bool, grid: bool) -> None:
    th = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "svg.fonttype": "path"})
    rng = np.random.default_rng(3)

    pts = make_data()
    ref = kde(pts, pts) - 1.0 / (len(pts) * 2.0 * np.pi * BW * BW)       # leave-one-out densities
    levels = np.percentile(ref, [2, 6, 12, 22, 35, 50, 66, 82])
    band = tuple(np.percentile(ref, [5, 15]))
    gx, gy = np.meshgrid(np.linspace(0, 1, 240), np.linspace(0, 1, 240))
    g = kde(pts, np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

    # trajectory from the dense core out along the lower-right flank, into unexplored space
    way = np.array([[0.36, 0.40], [0.46, 0.38], [0.56, 0.40], [0.66, 0.44], [0.77, 0.51]])
    half = np.array([0.075, 0.075])
    fracs = []
    for i in (2, 3):                     # current waypoint: the one whose box the density band crosses best
        c = way[i] + (np.random.default_rng(5).random((400, 2)) * 2 - 1) * half
        d = kde(pts, c)
        fracs.append(abs(((d >= band[0]) & (d <= band[1])).mean() - 0.35))
    cur = (2, 3)[int(np.argmin(fracs))]
    anchor = way[cur]
    cand, eligible, picked = plan(pts, anchor, half, band, np.random.default_rng(8))

    fig = plt.figure(figsize=(W / 10, H / 10), facecolor=th["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- lanes
    card(ax, 38, 47, 119, 42.5, th, edge="none", lw=0, fc=th["fwd_tint"], r=2.2, z=1)
    card(ax, 38, 1.8, 119, 42.2, th, edge="none", lw=0, fc=th["bwd_tint"], r=2.2, z=1)
    lane_label(ax, 40.5, 86.4, "GPU", "FORWARD JOB", th["fwd"], th)
    lane_label(ax, 40.5, 41.2, "CPU", "BACKWARD JOB", th["bwd"], th)
    ax.add_patch(FancyBboxPatch((128.6, 85.2), 2.4, 2.4, boxstyle="round,pad=0,rounding_size=0.5", fc=th["card"],
                                ec=th["al"], lw=2.0, zorder=5))
    ax.text(131.9, 86.4, "active-learning step", ha="left", va="center", fontsize=FS["small"], color=th["ink2"],
            zorder=6)

    # ---- (1) fine-tune EQDIFF
    card(ax, 41, 50, 28, 33, th, edge=th["fwd"], lw=1.1)
    step_title(ax, 43.6, 80.3, 1, "Fine-tune EQDIFF", th["fwd"], th)
    fy, fsz = 66.6, 6.6
    fxs = (42.6, 51.7, 60.8)
    for fx in fxs:
        card(ax, fx, fy, fsz, fsz, th, edge=th["edge"], lw=0.9, fc=th["frame"], r=0.8, z=3)
    c0 = (fxs[0] + fsz / 2 - 0.5, fy + fsz / 2 - 0.3)
    molecule(ax, *c0, 1.05, th, rot=8)
    molecule(ax, fxs[1] + fsz / 2 - 0.5, fy + fsz / 2 - 0.3, 1.05, th, rot=8, jitter=0.22, alpha=0.7, rng=rng)
    noise(ax, fxs[1], fy, fsz, th, rng, n=8)
    noise(ax, fxs[2], fy, fsz, th, rng, n=30)
    for fx, lab in zip(fxs, ("$x_0$", "$x_t$", "$x_T$")):
        ax.text(fx + fsz / 2, fy - 1.25, lab, ha="center", va="center", fontsize=FS["tiny"], color=th["ink2"],
                zorder=8)
    mid = (fxs[0] + fxs[2] + fsz) / 2
    arrow(ax, (fxs[0] + 1.6, 74.9), (fxs[2] + fsz - 1.6, 74.9), th, lw=1.3, ms=11)
    ax.text(mid, 76.5, "noising", ha="center", va="center", fontsize=FS["tiny"], color=th["ink2"], zorder=6)
    arrow(ax, (fxs[2] + fsz - 1.6, 63.0), (fxs[0] + 1.6, 63.0), th, lw=1.3, ms=11)
    ax.text(mid, 61.5, "denoising", ha="center", va="center", fontsize=FS["tiny"], color=th["ink2"], zorder=6)
    ax.text(42.6, 58.2, "weighted replay of all data", ha="left", va="center", fontsize=FS["small"],
            color=th["ink2"], zorder=6)
    shares, x0, bw = (0.186, 0.414, 0.40), 42.6, 24.8
    for share, col, lab in zip(shares, th["replay"], ("initial", "earlier", "newest")):
        w_ = share * bw
        card(ax, x0, 54.2, w_ - 0.35, 2.6, th, edge="none", lw=0, fc=col, r=0.5, z=4)
        ax.text(x0 + (w_ - 0.35) / 2, 52.2, lab, ha="center", va="center", fontsize=FS["tiny"], color=th["ink2"],
                zorder=6)
        x0 += w_

    # ---- (2) KDE target planning
    card(ax, 73, 50, 58, 33, th, edge=th["al"], lw=2.2)
    step_title(ax, 75.6, 80.3, 2, "Plan targets with KDE", th["fwd"], th)
    mx, my, msz = 77.0, 55.2, 22.0
    axp = map_axes(fig, mx, my, msz, msz)
    density(axp, gx, gy, g, levels, th)
    trajectory(axp, way, cur, th)
    axp.add_patch(Rectangle(anchor - half, 2 * half[0], 2 * half[1], fc="none", ec=th["al"], lw=1.4, ls="--",
                            zorder=7))
    axp.text(way[0, 0] - 0.03, way[0, 1] + 0.04, "START", ha="right", va="bottom", fontsize=9.5,
             fontweight="bold", color=th["ink2"], zorder=8)
    axp.text(way[-1, 0] + 0.02, way[-1, 1] - 0.05, "END", ha="left", va="top", fontsize=9.5, fontweight="bold",
             color=th["ink2"], zorder=8)
    axis_arrows(ax, mx, my, msz, msz, th)

    zx, zy, zs = 108.0, 57.0, 20.0
    lo_xy, hi_xy = anchor - half, anchor + half
    axz = map_axes(fig, zx, zy, zs, zs, xlim=(lo_xy[0], hi_xy[0]), ylim=(lo_xy[1], hi_xy[1]))
    zgx, zgy = np.meshgrid(np.linspace(lo_xy[0], hi_xy[0], 160), np.linspace(lo_xy[1], hi_xy[1], 160))
    zg = kde(pts, np.column_stack([zgx.ravel(), zgy.ravel()])).reshape(zgx.shape)
    density(axz, zgx, zgy, zg, levels, th)
    axz.contourf(zgx, zgy, zg, levels=[band[0], band[1]], colors=[th["band"]], alpha=0.92, zorder=3)
    axz.contour(zgx, zgy, zg, levels=[band[0], band[1]], colors=[th["al"]], linewidths=1.0,
                linestyles=[(0, (3, 2))], zorder=3)
    axz.scatter(cand[~eligible, 0], cand[~eligible, 1], s=9, c=th["muted"], edgecolors="none", zorder=4)
    axz.scatter(cand[eligible, 0], cand[eligible, 1], s=22, c=th["al_soft"], edgecolors=th["al"], linewidths=0.8,
                zorder=5)
    for i in picked:
        axz.plot([anchor[0], cand[i, 0]], [anchor[1], cand[i, 1]], color=th["al"], lw=1.1, ls=":", zorder=5)
    axz.scatter([anchor[0]], [anchor[1]], marker="*", s=330, c=th["ink"], edgecolors=th["card"], linewidths=1.3,
                zorder=7)
    axz.text(anchor[0] + 0.010, anchor[1] + 0.010, "$P^q$", ha="left", va="bottom", fontsize=FS["body"],
             color=th["ink"], zorder=8)
    for k, i in enumerate(picked):
        axz.scatter([cand[i, 0]], [cand[i, 1]], marker="*", s=360, c=th["al"], edgecolors=th["card"],
                    linewidths=1.3, zorder=7)
        dx = 0.010 if cand[i, 0] < hi_xy[0] - 0.035 else -0.010
        dy = 0.010 if cand[i, 1] < hi_xy[1] - 0.030 else -0.012
        axz.text(cand[i, 0] + dx, cand[i, 1] + dy, f"$T_{k + 1}$", ha="left" if dx > 0 else "right",
                 va="bottom" if dy > 0 else "top", fontsize=FS["body"], color=th["ink"], zorder=8)
    axz.add_patch(Rectangle((0, 0), 1, 1, transform=axz.transAxes, fc="none", ec=th["al"], lw=1.6, ls="--",
                            zorder=9, clip_on=False))
    for ya, yb in ((hi_xy[1], 1), (lo_xy[1], 0)):
        fig.add_artist(ConnectionPatch(xyA=(hi_xy[0], ya), coordsA=axp.transData, xyB=(0, yb),
                                       coordsB=axz.transAxes, color=th["muted"], lw=0.9, zorder=10))
    lx, ly = 77.0, 52.4
    items = [("*", th["ink"], 150, "waypoint $P^q$", 11), ("*", th["al"], 170, "KDE target", 10),
             ("s", th["band"], 60, "sparse band", 11), ("o", th["muted"], 14, "candidate", 9)]
    for mk, col, size, lab, nchar in items:
        ax.scatter([lx], [ly], marker=mk, s=size, c=col, edgecolors=th["al"] if mk == "s" else th["card"],
                   linewidths=0.8, zorder=8)
        ax.text(lx + 1.3, ly, lab, ha="left", va="center", fontsize=FS["tiny"], color=th["ink2"], zorder=8)
        lx += 2.7 + 0.92 * nchar

    # ---- (3) generate + dedup
    card(ax, 135, 50, 21, 33, th, edge=th["fwd"], lw=1.1)
    step_title(ax, 137.6, 80.3, 3, "Generate", th["fwd"], th, sub="at every target")
    rows = (71.2, 65.4, 59.6)
    for k, (yy, col) in enumerate(zip(rows, (th["ink"], th["al"], th["al"]))):
        ax.scatter([138.0], [yy], marker="*", s=230, c=col, edgecolors=th["card"], linewidths=1.0, zorder=7)
        arrow(ax, (139.6, yy), (142.4, yy), th, lw=1.2, ms=9)
        molecule(ax, 145.9, yy - 0.1, 0.82, th, rot=10 + 20 * k)
        dup = k == 2
        molecule(ax, 152.5, yy - 0.1, 0.82, th, rot=10 + 20 * (1 if dup else k) + (0 if dup else 30),
                 alpha=0.25 if dup else 1.0)
        if dup:
            ax.text(152.4, yy, "✗", ha="center", va="center", fontsize=20, fontweight="bold", color=th["al"],
                    zorder=9)
    ax.text(137.0, 53.4, "✗", ha="left", va="center", fontsize=13, fontweight="bold", color=th["al"], zorder=6)
    ax.text(139.0, 53.4, "duplicates removed", ha="left", va="center", fontsize=FS["small"], color=th["ink2"],
            zorder=6)

    arrow(ax, (69.0, 66.5), (73.0, 66.5), th)
    arrow(ax, (131.0, 66.5), (135.0, 66.5), th)
    arrow(ax, (145.0, 50.0), (145.0, 37.5), th)
    ax.text(146.2, 45.6, "molecules", ha="left", va="center", fontsize=FS["small"], color=th["ink2"], zorder=6)

    # ---- backward cards, right to left
    cw, cy0, ch = 22.5, 5.0, 32.5
    x4, x5, x6, x7 = 133.5, 107.0, 80.5, 54.0

    card(ax, x4, cy0, cw, ch, th, edge=th["bwd"], lw=1.1)                                   # (4) screen
    step_title(ax, x4 + 2.6, 35.0, 4, "Screen", th["bwd"], th, sub="QED · SA score")
    for yy, lab, keep_right, dot in ((24.5, "QED", True, x4 + 17.0), (17.0, "SA", False, x4 + 9.2)):
        ax.text(x4 + 1.8, yy, lab, ha="left", va="center", fontsize=FS["small"], color=th["ink"], zorder=6)
        x_a, x_b, x_t = x4 + 7.2, x4 + 20.4, x4 + 13.4
        seg = (x_t, x_b) if keep_right else (x_a, x_t)
        card(ax, seg[0], yy - 0.9, seg[1] - seg[0], 1.8, th, edge="none", lw=0, fc=th["bwd_soft"], r=0.9, z=4)
        ax.plot([x_a, x_b], [yy, yy], color=th["muted"], lw=1.2, zorder=5, solid_capstyle="round")
        ax.plot([x_t, x_t], [yy - 1.5, yy + 1.5], color=th["ink2"], lw=1.6, zorder=6)
        ax.scatter([dot], [yy], s=70, c=th["bwd"], edgecolors=th["card"], linewidths=1.2, zorder=7)
    ax.text(x4 + cw / 2, 10.0, "vs. initial mean", ha="center", va="center", fontsize=FS["small"],
            color=th["ink2"], zorder=6)

    card(ax, x5, cy0, cw, ch, th, edge=th["bwd"], lw=1.1)                                   # (5) relax
    step_title(ax, x5 + 2.6, 35.0, 5, "Relax", th["bwd"], th, sub="geometry optimization")
    for yy, lab in ((25.2, "EquiDTB (ML)"), (19.9, "DFTB+"), (14.6, "FHI-aims (DFT)")):
        pill(ax, x5 + cw / 2, yy, 18.0, lab, th, th["bwd"])

    card(ax, x6, cy0, cw, ch, th, edge=th["bwd"], lw=1.1)                                   # (6) properties
    step_title(ax, x6 + 2.6, 35.0, 6, "QM properties", th["bwd"], th)
    for yy, lab in ((27.4, "HOMO–LUMO gap"), (23.0, "MBD energy"), (18.6, "dipole, …")):
        ax.scatter([x6 + 2.6], [yy], s=26, c=th["bwd"], edgecolors="none", zorder=6)
        ax.text(x6 + 4.1, yy, lab, ha="left", va="center", fontsize=FS["small"], color=th["ink"], zorder=6)
    ax.text(x6 + cw / 2, 10.6, r"$\rightarrow\ (P_1, P_2, \ldots)$", ha="center", va="center",
            fontsize=FS["body"], color=th["ink2"], zorder=6)

    card(ax, x7, cy0, cw, ch, th, edge=th["al"], lw=2.2)                                    # (7) hit or miss
    step_title(ax, x7 + 2.6, 35.0, 7, "Hit or miss?", th["bwd"], th)
    bx, by, bhx, bhy = x7 + 11.0, 22.4, 4.6, 4.0
    ax.add_patch(Rectangle((bx - bhx, by - bhy), 2 * bhx, 2 * bhy, fc="none", ec=th["ink2"], lw=1.2, ls="--",
                           zorder=5))
    ax.text(bx - bhx, by + bhy + 0.9, "tolerance", ha="left", va="center", fontsize=FS["tiny"], color=th["ink2"],
            zorder=6)
    ax.scatter([bx], [by], marker="*", s=260, c=th["al"], edgecolors=th["card"], linewidths=1.0, zorder=6)
    for px, py in ((bx - 2.3, by + 1.7), (bx + 2.2, by - 2.3), (bx + 1.5, by + 2.5)):
        ax.text(px, py, "✓", ha="center", va="center", fontsize=15, fontweight="bold", color=th["good"], zorder=7)
    for px, py in ((bx + 7.6, by + 0.6), (bx - 7.2, by - 5.6)):
        ax.text(px, py, "✗", ha="center", va="center", fontsize=15, fontweight="bold", color=th["bad"], zorder=7)
    ax.text(x7 + 2.0, 12.0, "✓", ha="left", va="center", fontsize=13, fontweight="bold", color=th["good"], zorder=6)
    ax.text(x7 + 4.0, 12.0, "kept", ha="left", va="center", fontsize=FS["small"], color=th["ink2"], zorder=6)
    ax.text(x7 + 11.6, 12.0, "✗", ha="left", va="center", fontsize=13, fontweight="bold", color=th["bad"], zorder=6)
    ax.text(x7 + 13.6, 12.0, "dropped", ha="left", va="center", fontsize=FS["small"], color=th["ink2"], zorder=6)
    ax.text(x7 + cw / 2, 8.2, "vs. its own target", ha="center", va="center", fontsize=FS["small"],
            color=th["ink2"], zorder=6)

    for xa, xb in ((x4, x5 + cw), (x5, x6 + cw), (x6, x7 + cw)):
        arrow(ax, (xa, 21.25), (xb, 21.25), th)

    # ---- left column: property space, datasets, loop decision
    card(ax, 2.5, 22.0, 32.5, 39.0, th, edge=th["edge"], lw=1.1)
    ax.text(4.8, 58.3, "Property space", ha="left", va="center", fontsize=FS["title"], fontweight="bold",
            color=th["ink"], zorder=6)
    px0, py0, psz = 6.5, 29.5, 24.5
    axl = map_axes(fig, px0, py0, psz, psz)
    density(axl, gx, gy, g, levels, th)
    trajectory(axl, way, cur, th, faint=True)
    arng, added = np.random.default_rng(21), []
    for i in range(cur):
        c = way[i] + (arng.random((80, 2)) * 2 - 1) * half * 1.15
        d = kde(pts, c)
        added.append(c[(d >= band[0] * 0.5) & (d <= band[1] * 1.8)][:11])
    added = np.vstack(added)
    axl.scatter(added[:, 0], added[:, 1], s=30, c=th["bwd"], edgecolors=th["card"], linewidths=0.7, zorder=7)
    axis_arrows(ax, px0, py0, psz, psz, th)
    ax.scatter([8.2], [25.2], s=40, c=th["bwd"], edgecolors=th["card"], linewidths=0.6, zorder=6)
    ax.text(9.5, 25.2, "added molecules", ha="left", va="center", fontsize=FS["tiny"], color=th["ink2"], zorder=6)

    cylinder(ax, 5.0, 3.6, 14.0, 13.5, "Initial\ndataset", th)
    arrow(ax, (12.0, 17.2), (12.0, 22.0), th)
    cylinder(ax, 2.5, 69.4, 12.0, 13.2, "Final\ndataset", th, fs=12.5)
    diamond(ax, 27.0, 76.0, 8.5, 6.6, r"$q > N_\mathrm{iter}$ ?", th)
    arrow(ax, (27.0, 61.0), (27.0, 69.4), th)
    arrow(ax, (18.5, 76.0), (14.6, 76.0), th)
    ax.text(16.6, 77.6, "yes", ha="center", va="center", fontsize=FS["tiny"], color=th["ink2"], zorder=6)
    arrow(ax, (35.5, 76.0), (41.0, 76.0), th)
    ax.text(38.2, 77.6, "no", ha="center", va="center", fontsize=FS["tiny"], color=th["ink2"], zorder=6)
    arrow(ax, (x7, 30.5), (35.0, 30.5), th)
    ax.text((x7 + 35.0) / 2, 32.4, "add hits", ha="center", va="center", fontsize=FS["small"], fontweight="bold",
            color=th["ink"], zorder=6)
    ax.text((x7 + 35.0) / 2, 28.4, r"$q \leftarrow q + 1$", ha="center", va="center", fontsize=FS["small"],
            color=th["ink2"], zorder=6)

    if grid:
        for gxl in range(0, int(W) + 1, 5):
            ax.axvline(gxl, color="#ff00ff", lw=0.3, alpha=0.4, zorder=20)
            ax.text(gxl, 0.5, str(gxl), fontsize=6, color="#ff00ff", zorder=21)
        for gyl in range(0, int(H) + 1, 5):
            ax.axhline(gyl, color="#ff00ff", lw=0.3, alpha=0.4, zorder=20)
            ax.text(0.3, gyl, str(gyl), fontsize=6, color="#ff00ff", zorder=21)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=DPI, facecolor=th["bg"])
    if svg:
        fig.savefig(out.with_suffix(".svg"), facecolor=th["bg"])
    plt.close(fig)
    print(f"{theme}: current waypoint {cur}, eligible {int(eligible.sum())}/{len(cand)}, picked {picked} -> {out}")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=here)
    p.add_argument("--svg", action="store_true", help="also write SVG files")
    p.add_argument("--grid", action="store_true", help="overlay a layout grid (for editing)")
    p.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    args = p.parse_args()
    for theme in (["light", "dark"] if args.theme == "both" else [args.theme]):
        name = "qalpa-workflow" if theme == "light" else "qalpa-workflow-dark"
        draw(theme, args.out_dir / name, args.svg, args.grid)


if __name__ == "__main__":
    main()
