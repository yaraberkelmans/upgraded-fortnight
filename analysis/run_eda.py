#!/usr/bin/env python3
"""
Run-level EDA: full time-series output for two representative runs
(similarity_threshold ≈ 0.4 and ≈ 0.8) plus warmup/entropy comparison.

Run from project root:
    python analysis/run_eda.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR  = Path("data") / "seed_43"
PLOTS_DIR = Path("plots") / "run_eda"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = ["similarity_threshold", "fight_threshold", "hawk_dove_C", "police_density"]
COLORS = {"sim=0.4": "steelblue", "sim=0.8": "darkorange"}

# ── Find representative runs ────────────────────────────────────────────────────
rr    = np.load(DATA_DIR / "run_results.npy")
rr_df = pd.DataFrame(rr)
sim   = rr_df["similarity_threshold"].values

runs = {}
for label, target in [("sim=0.4", 0.4), ("sim=0.8", 0.8)]:
    run_id = int(np.argmin(np.abs(sim - target)))
    row    = rr_df.iloc[run_id]
    ts     = pd.DataFrame(np.load(DATA_DIR / "runs"     / f"run_{run_id:04d}.npy"))
    ov     = np.load(DATA_DIR / "overview" / f"overview_{run_id:04d}.npy")[0]

    total_fans          = ts["home"] + ts["away"]
    ts["home_fraction"] = ts["home"] / total_fans
    ts["entropy_norm"]  = ts["spatial_entropy_local"] / np.log(2)   # 0=segregated, 1=fully mixed

    runs[label] = dict(run_id=run_id, color=COLORS[label], ts=ts, overview=ov,
                       params={p: row[p] for p in PARAMS})

    print(f"{label}  run={run_id}  "
          f"sim={row.similarity_threshold:.3f}  fight_thr={row.fight_threshold:.3f}  "
          f"C={row.hawk_dove_C:.2f}  police={row.police_density:.3f}  "
          f"warmup={int(ov['warmup_steps'])} steps")

steps = np.arange(100)


# ── Layout helpers ──────────────────────────────────────────────────────────────
def param_subtitle(r):
    p  = r["params"]
    ov = r["overview"]
    return (f"run {r['run_id']}  |  "
            f"sim_thr={p['similarity_threshold']:.2f}  "
            f"fight_thr={p['fight_threshold']:.2f}  "
            f"C={p['hawk_dove_C']:.1f}  "
            f"police={p['police_density']:.2f}  "
            f"|  warmup={int(ov['warmup_steps'])} steps")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Full time-series, 5 groups × 2 runs side-by-side
# ══════════════════════════════════════════════════════════════════════════════
GROUPS = [
    {
        "title": "Satisfaction & Balance",
        "ylabel": "Fraction",
        "ylim":   (0, 1),
        "series": [
            ("home_fraction",  "Home fraction",  {}),
            ("happy_fraction", "Happy fraction", {"linestyle": "--"}),
        ],
    },
    {
        "title": "Fan Movement (Schelling)",
        "ylabel": "Moves per step",
        "ylim":   None,
        "series": [
            ("moves", "Moves", {}),
        ],
    },
    {
        "title": "Violence & Arrests",
        "ylabel": "Count",
        "ylim":   None,
        "series": [
            ("fighting",      "Fighting fans", {}),
            ("arrests_step",  "Arrests / step", {"linestyle": "--"}),
        ],
    },
    {
        "title": "Perceived Probabilities",
        "ylabel": "Probability",
        "ylim":   (0, 1),
        "series": [
            ("average_win_probability",    "Win probability",    {}),
            ("average_arrest_probability", "Arrest probability", {"linestyle": "--"}),
        ],
    },
    {
        "title": "Spatial Segregation",
        "ylabel": "Value (0=segregated, 1=mixed/similar)",
        "ylim":   (0, 1),
        "series": [
            ("entropy_norm",       "Entropy (normalised)", {}),
            ("average_similarity", "Avg similarity",       {"linestyle": "--"}),
        ],
    },
]

labels  = list(runs.keys())
n_grp   = len(GROUPS)
fig, axes = plt.subplots(n_grp, 2, figsize=(15, 3.5 * n_grp), sharey="row")

for col, label in enumerate(labels):
    r  = runs[label]
    ts = r["ts"]
    c  = r["color"]

    axes[0, col].set_title(param_subtitle(r), fontsize=8, pad=6)

    for row_i, grp in enumerate(GROUPS):
        ax = axes[row_i, col]
        for key, name, kwargs in grp["series"]:
            ax.plot(steps, ts[key], color=c, lw=1.8, label=name, **kwargs)
        if grp["ylim"]:
            ax.set_ylim(*grp["ylim"])
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("Measurement step" if row_i == n_grp - 1 else "")
        if col == 0:
            ax.set_ylabel(f"{grp['title']}\n{grp['ylabel']}", fontsize=8)

# Column headers
for col, label in enumerate(labels):
    axes[0, col].annotate(label, xy=(0.5, 1.18), xycoords="axes fraction",
                          ha="center", fontsize=12, fontweight="bold",
                          color=runs[label]["color"])

fig.suptitle("Measurement-window time series  (seed 43)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "01_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved: 01_timeseries.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Warmup duration + entropy at key time points
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax_warm, ax_ent) = plt.subplots(1, 2, figsize=(12, 5))

# ── Left: warmup duration ─────────────────────────────────────────────────────
for i, (label, r) in enumerate(runs.items()):
    w = int(r["overview"]["warmup_steps"])
    ax_warm.bar(i, w, color=r["color"], alpha=0.8, label=label, width=0.5)
    ax_warm.text(i, w + 1, str(w), ha="center", va="bottom", fontsize=11,
                 color=r["color"], fontweight="bold")

ax_warm.set_xticks(range(len(runs)))
ax_warm.set_xticklabels(labels, fontsize=11)
ax_warm.set_ylabel("Steps until convergence")
ax_warm.set_title("Warmup Duration\n(steps until fine-entropy CV < threshold)")
ax_warm.set_ylim(0, max(int(r["overview"]["warmup_steps"]) for r in runs.values()) * 1.25)

# ── Right: entropy at three time points ──────────────────────────────────────
EPOINTS = ["warmup_entropy", "start_measurement_entropy", "end_measurement_entropy"]
ELABELS = ["End of warmup", "Start of measurement\n(after burn-in)", "End of measurement"]
x = np.arange(3)

for label, r in runs.items():
    ov   = r["overview"]
    vals = [float(ov[ep]) for ep in EPOINTS]
    ax_ent.plot(x, vals, "o-", color=r["color"], lw=2.5, ms=9, label=label)
    for xi, v in zip(x, vals):
        ax_ent.annotate(f"{v:.3f}", (xi, v),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=r["color"])

ax_ent.axhline(np.log(2), color="gray", lw=1, linestyle=":",
               label=f"max entropy ln(2) ≈ {np.log(2):.3f}")
ax_ent.set_xticks(x)
ax_ent.set_xticklabels(ELABELS, fontsize=9)
ax_ent.set_ylabel("Spatial entropy  (0 = fully segregated)")
ax_ent.set_title("Spatial Entropy at Key Time Points")
ax_ent.set_ylim(0, np.log(2) * 1.2)
ax_ent.legend(fontsize=9)

fig.suptitle("Warmup & Entropy Comparison  (seed 43)", fontsize=13)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "02_warmup_entropy.png", dpi=150)
plt.close(fig)
print("saved: 02_warmup_entropy.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Overview stats vs similarity_threshold (all 5 seeds)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading all seeds for overview comparison…")

SEEDS      = [43, 44, 45, 46, 47]
RAW_COLS   = ["warmup_steps", "warmup_entropy", "start_measurement_entropy", "end_measurement_entropy"]

frames = []
for seed in SEEDS:
    rr = pd.DataFrame(np.load(Path("data") / f"seed_{seed}" / "run_results.npy"))
    ov = pd.DataFrame(np.load(Path("data") / f"seed_{seed}" / "run_overview.npy"))
    merged = rr[["sample_id", "similarity_threshold"]].merge(
        ov[["run_id"] + RAW_COLS],
        left_on="sample_id", right_on="run_id", how="inner"
    )
    frames.append(merged)

all_df = pd.concat(frames, ignore_index=True)
all_df["entropy_drop"] = all_df["warmup_entropy"] - all_df["start_measurement_entropy"]
print(f"  {len(all_df):,} rows across {len(SEEDS)} seeds")

N_BINS     = 30
all_df["sim_bin"] = pd.cut(all_df["similarity_threshold"], bins=N_BINS)
grp        = all_df.groupby("sim_bin", observed=True)
bin_center = grp["similarity_threshold"].mean().values
x_mask     = ~np.isnan(bin_center)

def binned(col):
    y = grp[col].mean().values
    s = grp[col].std().values
    return y, s

def plot_band(ax, x, y, s, color, label):
    m = x_mask & ~np.isnan(y)
    ax.plot(x[m], y[m], color=color, lw=2, label=label)
    ax.fill_between(x[m], y[m] - s[m], y[m] + s[m], alpha=0.22, color=color)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax1, ax2, ax3, ax4 = axes.flat

# ── Panel 1: Warmup steps ──────────────────────────────────────────────────────
y, s = binned("warmup_steps")
plot_band(ax1, bin_center, y, s, "steelblue", "Mean ± 1 SD")
ax1.set_title("Warmup Steps")
ax1.set_ylabel("Steps")
ax1.legend(fontsize=9)

# ── Panel 2: Warmup entropy + Start-of-measurement entropy (overlapped) ────────
for col, color, label in [
    ("warmup_entropy",            "steelblue",  "After warmup"),
    ("start_measurement_entropy", "darkorange", "Start of measurement"),
]:
    y, s = binned(col)
    plot_band(ax2, bin_center, y, s, color, label)
ax2.set_title("Entropy: After Warmup vs Start of Measurement")
ax2.set_ylabel("Spatial Entropy")
ax2.axhline(np.log(2), color="gray", lw=1, ls=":", label=f"max ln(2)≈{np.log(2):.3f}")
ax2.legend(fontsize=9)

# ── Panel 3: End-of-measurement entropy ───────────────────────────────────────
y, s = binned("end_measurement_entropy")
plot_band(ax3, bin_center, y, s, "steelblue", "Mean ± 1 SD")
ax3.set_title("End-of-Measurement Entropy")
ax3.set_ylabel("Spatial Entropy")
ax3.axhline(np.log(2), color="gray", lw=1, ls=":")
ax3.legend(fontsize=9)

# ── Panel 4: Entropy drop  (warmup − start-of-measurement) ───────────────────
y, s = binned("entropy_drop")
plot_band(ax4, bin_center, y, s, "mediumpurple", "Mean ± 1 SD")
ax4.axhline(0, color="gray", lw=1, ls=":")
ax4.set_title("Entropy Drop: After Warmup − Start of Measurement\n(positive = grid segregates further during burn-in)")
ax4.set_ylabel("Entropy difference")
ax4.legend(fontsize=9)

for ax in axes.flat:
    ax.set_xlabel("Similarity Threshold")

fig.suptitle(
    f"Overview Statistics vs Similarity Threshold\n"
    f"({len(SEEDS)} seeds × 3,072 runs = {len(all_df):,} data points, {N_BINS} bins)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "03_overview_vs_sim_threshold.png", dpi=150)
plt.close(fig)
print("saved: 03_overview_vs_sim_threshold.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — 3-D surfaces: each overview panel vs sim_threshold × other param
# ══════════════════════════════════════════════════════════════════════════════
print("Loading full 4-param data for 3-D surfaces…")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

frames4 = []
for seed in SEEDS:
    rr4 = pd.DataFrame(np.load(Path("data") / f"seed_{seed}" / "run_results.npy"))
    ov4 = pd.DataFrame(np.load(Path("data") / f"seed_{seed}" / "run_overview.npy"))
    m4  = rr4[["sample_id", "similarity_threshold",
                "fight_threshold", "hawk_dove_C", "police_density"]].merge(
        ov4[["run_id"] + RAW_COLS],
        left_on="sample_id", right_on="run_id", how="inner"
    )
    frames4.append(m4)

full_df = pd.concat(frames4, ignore_index=True)
full_df["entropy_drop"] = full_df["warmup_entropy"] - full_df["start_measurement_entropy"]
print(f"  {len(full_df):,} rows")

N3 = 15  # bins per axis for 3-D surfaces


def surface_2d(col, xp, yp, n=N3):
    """Bin full_df by xp × yp, return mean(col) as a (n,n) surface meshgrid."""
    xe  = np.linspace(full_df[xp].min(), full_df[xp].max(), n + 1)
    ye  = np.linspace(full_df[yp].min(), full_df[yp].max(), n + 1)
    xc  = 0.5 * (xe[:-1] + xe[1:])
    yc  = 0.5 * (ye[:-1] + ye[1:])
    xi  = np.clip(np.digitize(full_df[xp].values, xe) - 1, 0, n - 1)
    yi  = np.clip(np.digitize(full_df[yp].values, ye) - 1, 0, n - 1)
    Z_s = np.zeros((n, n))
    Z_n = np.zeros((n, n), dtype=int)
    np.add.at(Z_s, (xi, yi), full_df[col].values)
    np.add.at(Z_n, (xi, yi), 1)
    Z = np.where(Z_n > 0, Z_s / Z_n, np.nan)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return X, Y, Z


Y_PARAMS_3D = [
    ("fight_threshold", "Fight Threshold"),
    ("hawk_dove_C",     "Hawk-Dove C"),
    ("police_density",  "Police Density"),
]

PANELS_3D = [
    ("warmup_steps",            "Warmup Steps",                                "viridis"),
    (None,                       "Entropy: After Warmup\nvs Start of Meas.",   None),
    ("end_measurement_entropy", "End-of-Measurement Entropy",                  "plasma"),
    ("entropy_drop",            "Entropy Drop\n(Warmup − Start of Meas.)",     "RdBu_r"),
]

for y_param, y_label in Y_PARAMS_3D:
    print(f"  3-D surfaces × {y_label}…")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Overview Metrics  vs  Similarity Threshold  ×  {y_label}\n"
        f"({len(SEEDS)} seeds × 3,072 runs = {len(full_df):,} data points, {N3}×{N3} bins)",
        fontsize=12,
    )

    for pos, (metric, title, cmap) in enumerate(PANELS_3D, start=1):
        ax = fig.add_subplot(2, 2, pos, projection="3d")
        ax.view_init(elev=28, azim=-55)

        if metric is None:
            # Panel 2: overlay two solid-colour surfaces
            for col, color, lbl in [
                ("warmup_entropy",            "steelblue",  "After warmup"),
                ("start_measurement_entropy", "darkorange", "Start of meas."),
            ]:
                X, Y, Z = surface_2d(col, y_param, "similarity_threshold")
                ax.plot_surface(X, Y, Z, color=color, alpha=0.60, linewidth=0)
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(color="steelblue",  alpha=0.8, label="After warmup"),
                Patch(color="darkorange", alpha=0.8, label="Start of meas."),
            ], fontsize=8, loc="upper left")
        else:
            X, Y, Z = surface_2d(metric, y_param, "similarity_threshold")
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.90, linewidth=0)
            fig.colorbar(surf, ax=ax, shrink=0.38, pad=0.08)

        ax.set_xlabel(y_label,                 labelpad=6, fontsize=9)
        ax.set_ylabel("Similarity\nThreshold", labelpad=8, fontsize=9)
        ax.set_title(title, fontsize=10)

    fname = f"04_3d_{y_param}.png"
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"    saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — 3-D surfaces: mean_fighting vs sim_threshold × each other param
# ══════════════════════════════════════════════════════════════════════════════
print("Building mean_fighting 3-D surfaces…")

# Add mean_fighting from run_results to full_df
frames5 = []
for seed in SEEDS:
    rr5 = pd.DataFrame(np.load(Path("data") / f"seed_{seed}" / "run_results.npy"))
    frames5.append(rr5[["sample_id", "similarity_threshold",
                         "fight_threshold", "hawk_dove_C", "police_density",
                         "mean_fighting", "std_fighting"]])

fights_df = pd.concat(frames5, ignore_index=True)

def surface_2d_df(df, col, xp, yp, n=N3, agg="mean"):
    """
    agg='mean' : mean of col per (xp_bin, yp_bin)
    agg='std'  : sample std of col per bin (cross-run variability within each cell)
    """
    xe  = np.linspace(df[xp].min(), df[xp].max(), n + 1)
    ye  = np.linspace(df[yp].min(), df[yp].max(), n + 1)
    xc  = 0.5 * (xe[:-1] + xe[1:])
    yc  = 0.5 * (ye[:-1] + ye[1:])
    xi  = np.clip(np.digitize(df[xp].values, xe) - 1, 0, n - 1)
    yi  = np.clip(np.digitize(df[yp].values, ye) - 1, 0, n - 1)
    vals = df[col].values.astype(float)
    Z_s  = np.zeros((n, n))
    Z_s2 = np.zeros((n, n))
    Z_n  = np.zeros((n, n), dtype=int)
    np.add.at(Z_s,  (xi, yi), vals)
    np.add.at(Z_s2, (xi, yi), vals ** 2)
    np.add.at(Z_n,  (xi, yi), 1)
    Z_mean = np.where(Z_n > 0, Z_s / Z_n, np.nan)
    if agg == "mean":
        Z = Z_mean
    else:
        # sample variance: (sum_x2 - n*mean^2) / (n-1)
        Z = np.where(Z_n > 1,
                     np.sqrt(np.maximum(0, (Z_s2 - Z_n * Z_mean**2) / (Z_n - 1))),
                     np.nan)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return X, Y, Z

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "Fighting Fans  vs  Similarity Threshold  ×  Each Input Parameter\n"
    f"({len(SEEDS)} seeds × 3,072 runs = {len(fights_df):,} data points, {N3}×{N3} bins)",
    fontsize=12,
)

for col_i, (y_param, y_label) in enumerate(Y_PARAMS_3D):
    Xm, Ym, Zm = surface_2d_df(fights_df, "mean_fighting", y_param, "similarity_threshold")
    Xs, Ys, Zs = surface_2d_df(fights_df, "std_fighting",  y_param, "similarity_threshold")

    for row_i, (Z, zlabel, title) in enumerate([
        (Zm, "Mean fighting fans / step",  f"Mean  —  {y_label}"),
        (Zs, "Std fighting fans / step",   f"Std  —  {y_label}"),
    ]):
        pos = row_i * 3 + col_i + 1
        ax  = fig.add_subplot(2, 3, pos, projection="3d")
        surf = ax.plot_surface(Xm if row_i == 0 else Xs,
                               Ym if row_i == 0 else Ys,
                               Z, cmap="inferno", alpha=0.90, linewidth=0)
        fig.colorbar(surf, ax=ax, shrink=0.38, pad=0.08, label=zlabel)
        ax.view_init(elev=28, azim=-55)
        ax.set_xlabel(y_label,                labelpad=6, fontsize=9)
        ax.set_ylabel("Similarity Threshold", labelpad=6, fontsize=9)
        ax.set_title(title, fontsize=10)

fname = "05_fighting_3d.png"
fig.tight_layout()
fig.savefig(PLOTS_DIR / fname, dpi=150)
plt.close(fig)
print(f"  saved: {fname}")

# Split version: one figure per half of similarity_threshold
SIM_SPLITS = [
    ("low",  fights_df[fights_df["similarity_threshold"] <  0.5], "Similarity Threshold < 0.5"),
    ("high", fights_df[fights_df["similarity_threshold"] >= 0.5], "Similarity Threshold >= 0.5"),
]

for split_name, split_df, split_label in SIM_SPLITS:
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Fighting Fans  vs  {split_label}  ×  Each Input Parameter\n"
        f"({len(split_df):,} data points, {N3}×{N3} bins)",
        fontsize=12,
    )

    for col_i, (y_param, y_label) in enumerate(Y_PARAMS_3D):
        Xm, Ym, Zm = surface_2d_df(split_df, "mean_fighting", y_param, "similarity_threshold")
        Xs, Ys, Zs = surface_2d_df(split_df, "std_fighting",  y_param, "similarity_threshold")

        for row_i, (X, Y, Z, zlabel, title) in enumerate([
            (Xm, Ym, Zm, "Mean fighting fans / step", f"Mean  —  {y_label}"),
            (Xs, Ys, Zs, "Std fighting fans / step",  f"Std  —  {y_label}"),
        ]):
            pos  = row_i * 3 + col_i + 1
            ax   = fig.add_subplot(2, 3, pos, projection="3d")
            surf = ax.plot_surface(X, Y, Z, cmap="inferno", alpha=0.90, linewidth=0)
            fig.colorbar(surf, ax=ax, shrink=0.38, pad=0.08, label=zlabel)
            ax.view_init(elev=28, azim=-55)
            ax.set_xlabel(y_label,                labelpad=6, fontsize=9)
            ax.set_ylabel("Similarity Threshold", labelpad=6, fontsize=9)
            ax.set_title(title, fontsize=10)

    fname = f"05_fighting_3d_{split_name}_sim.png"
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — 2×3 surfaces: arrests/step, happy fraction, unhappy fraction
# ══════════════════════════════════════════════════════════════════════════════
print("Building arrests / happy / unhappy 3-D surfaces…")

frames6 = []
for seed in SEEDS:
    rr6 = pd.DataFrame(np.load(Path("data") / f"seed_{seed}" / "run_results.npy"))
    frames6.append(rr6[["sample_id", "similarity_threshold",
                         "fight_threshold", "hawk_dove_C", "police_density",
                         "mean_arrests_per_step", "mean_happy_fraction",
                         "mean_fighting"]])

metrics_df = pd.concat(frames6, ignore_index=True)
metrics_df["mean_unhappy_fraction"] = 1.0 - metrics_df["mean_happy_fraction"]

METRICS_6 = [
    ("mean_arrests_per_step",  "Mean Arrests / Step",    "magma"),
    ("mean_happy_fraction",    "Mean Happy Fraction",     "YlGn"),
    ("mean_unhappy_fraction",  "Mean Unhappy Fraction",   "YlOrRd"),
]

for col, col_label, cmap in METRICS_6:
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"{col_label}  vs  Similarity Threshold  ×  Each Input Parameter\n"
        f"({len(SEEDS)} seeds × 3,072 runs = {len(metrics_df):,} data points, {N3}×{N3} bins)",
        fontsize=12,
    )

    for col_i, (y_param, y_label) in enumerate(Y_PARAMS_3D):
        for row_i, (agg, zlabel, row_title) in enumerate([
            ("mean", col_label,             f"Mean  —  {y_label}"),
            ("std",  f"Std  {col_label}",   f"Std  —  {y_label}"),
        ]):
            X, Y, Z = surface_2d_df(metrics_df, col, y_param, "similarity_threshold", agg=agg)
            pos  = row_i * 3 + col_i + 1
            ax   = fig.add_subplot(2, 3, pos, projection="3d")
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.90, linewidth=0)
            fig.colorbar(surf, ax=ax, shrink=0.38, pad=0.08, label=zlabel)
            ax.view_init(elev=28, azim=-55)
            ax.set_xlabel(y_label,                labelpad=6, fontsize=9)
            ax.set_ylabel("Similarity Threshold", labelpad=6, fontsize=9)
            ax.set_title(row_title, fontsize=10)

    fname = f"06_3d_{col}.png"
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — 2×3 surface: mean_fighting − mean_arrests_per_step
# ══════════════════════════════════════════════════════════════════════════════
print("Building (mean_fighting - mean_arrests_per_step) 3-D surfaces…")

metrics_df["fight_minus_arrests"] = metrics_df["mean_fighting"] - metrics_df["mean_arrests_per_step"]

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "Mean Fighting  −  Mean Arrests/Step  vs  Similarity Threshold  ×  Each Input Parameter\n"
    f"({len(SEEDS)} seeds × 3,072 runs = {len(metrics_df):,} data points, {N3}×{N3} bins)",
    fontsize=12,
)

for col_i, (y_param, y_label) in enumerate(Y_PARAMS_3D):
    for row_i, (agg, zlabel, row_title) in enumerate([
        ("mean", "Mean (fighting − arrests/step)", f"Mean  —  {y_label}"),
        ("std",  "Std (fighting − arrests/step)",  f"Std  —  {y_label}"),
    ]):
        X, Y, Z = surface_2d_df(metrics_df, "fight_minus_arrests", y_param, "similarity_threshold", agg=agg)
        pos  = row_i * 3 + col_i + 1
        ax   = fig.add_subplot(2, 3, pos, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="RdBu_r", alpha=0.90, linewidth=0)
        fig.colorbar(surf, ax=ax, shrink=0.38, pad=0.08, label=zlabel)
        ax.view_init(elev=28, azim=-55)
        ax.set_xlabel(y_label,                labelpad=6, fontsize=9)
        ax.set_ylabel("Similarity Threshold", labelpad=6, fontsize=9)
        ax.set_title(row_title, fontsize=10)

fname = "07_3d_fight_minus_arrests.png"
fig.tight_layout()
fig.savefig(PLOTS_DIR / fname, dpi=150)
plt.close(fig)
print(f"  saved: {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 8 & 9 — Entropy drop + mean fighting: hawk_dove_C × fight_threshold
#                 Three versions: sim < 0.5 | sim >= 0.5 | full data
# ══════════════════════════════════════════════════════════════════════════════
print("Building hawk_dove_C × fight_threshold surfaces (1×3 grids)…")

SIM_SUBSETS = [
    (full_df[full_df["similarity_threshold"] <  0.5], fights_df[fights_df["similarity_threshold"] <  0.5], "sim < 0.5"),
    (full_df[full_df["similarity_threshold"] >= 0.5], fights_df[fights_df["similarity_threshold"] >= 0.5], "sim >= 0.5"),
    (full_df,                                          fights_df,                                           "full data"),
]

# Figure 8 — entropy drop, 1×3
fig8 = plt.figure(figsize=(24, 7))
fig8.suptitle(
    "Entropy Drop (Warmup − Start of Meas.)  vs  Hawk-Dove C  ×  Fight Threshold",
    fontsize=13,
)
for col_i, (ov_sub, _, label) in enumerate(SIM_SUBSETS, start=1):
    X, Y, Z = surface_2d_df(ov_sub, "entropy_drop", "hawk_dove_C", "fight_threshold")
    ax   = fig8.add_subplot(1, 3, col_i, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="RdBu_r", alpha=0.90, linewidth=0)
    fig8.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label="Entropy drop")
    ax.view_init(elev=28, azim=-55)
    ax.set_xlabel("Hawk-Dove C",     labelpad=6, fontsize=9)
    ax.set_ylabel("Fight Threshold", labelpad=8, fontsize=9)
    ax.set_title(label, fontsize=11)

fig8.tight_layout()
fig8.savefig(PLOTS_DIR / "08_entropy_drop_hawkdove_x_fightthreshold.png", dpi=150)
plt.close(fig8)
print("  saved: 08_entropy_drop_hawkdove_x_fightthreshold.png")

# Figure 9 — mean fighting, 1×3
fig9 = plt.figure(figsize=(24, 7))
fig9.suptitle(
    "Mean Fighting Fans  vs  Hawk-Dove C  ×  Fight Threshold",
    fontsize=13,
)
for col_i, (_, fi_sub, label) in enumerate(SIM_SUBSETS, start=1):
    Xm, Ym, Zm = surface_2d_df(fi_sub, "mean_fighting", "hawk_dove_C", "fight_threshold")
    ax   = fig9.add_subplot(1, 3, col_i, projection="3d")
    surf = ax.plot_surface(Xm, Ym, Zm, cmap="inferno", alpha=0.90, linewidth=0)
    fig9.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label="Mean fighting fans / step")
    ax.view_init(elev=28, azim=-55)
    ax.set_xlabel("Hawk-Dove C",     labelpad=6, fontsize=9)
    ax.set_ylabel("Fight Threshold", labelpad=8, fontsize=9)
    ax.set_title(label, fontsize=11)

fig9.tight_layout()
fig9.savefig(PLOTS_DIR / "09_fighting_hawkdove_x_fightthreshold.png", dpi=150)
plt.close(fig9)
print("  saved: 09_fighting_hawkdove_x_fightthreshold.png")

print(f"\nAll plots saved to {PLOTS_DIR.resolve()}")
