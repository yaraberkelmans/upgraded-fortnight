"""Self-contained plots from the downloaded Snellius Sobol data (matplotlib only).

Reads each seed_*/run_results.npy, recomputes first-order Sobol internally
(averaging each QoI over the replicate seeds), and writes individual PNGs.

Figures:
  sobol_<qoi>.png      S1/ST bars (bootstrap CI) + per-seed ST dots
  scatter_<qoi>.png    QoI vs each param + binned mean + R^2 (variance explained
                       by the binned conditional mean; nonlinear analog of S1)
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.analyze import sobol

QOIS = ["mean_fighting_fraction", "mean_arrests_per_step", "mean_spatial_entropy_local"]

FALLBACK_PROBLEM = {
    "num_vars": 4,
    "names": ["similarity_threshold", "fight_threshold", "hawk_dove_C", "police_density"],
    "bounds": [[0.05, 0.90], [-0.35, 0.45], [0.10, 19.00], [0.01, 0.15]],
}


def find_seed_dirs(data_dir):
    dirs = sorted(glob.glob(os.path.join(data_dir, "seed_*")))
    if not dirs:
        dirs = [data_dir]
    return [d for d in dirs if os.path.exists(os.path.join(d, "run_results.npy"))]


def load_problem(seed_dirs):
    for d in seed_dirs:
        meta = os.path.join(d, "metadata.json")
        if os.path.exists(meta):
            p = json.load(open(meta))["problem"]
            return {"num_vars": int(p["num_vars"]), "names": list(p["names"]),
                    "bounds": [list(b) for b in p["bounds"]]}
    return FALLBACK_PROBLEM


def binned_mean(x, y, bins=12):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    cx, cy = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 3:
            cx.append(x[m].mean())
            cy.append(y[m].mean())
    return np.array(cx), np.array(cy)


def variance_explained(x, y, bins=12):
    """Fraction of Var(y) explained by the conditional mean over x-bins (eta^2)."""
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    gm = y.mean()
    ss_tot = np.sum((y - gm) ** 2)
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ss_between += m.sum() * (y[m].mean() - gm) ** 2
    return ss_between / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join("plots/sobol")
    os.makedirs(out_dir, exist_ok=True)

    seed_dirs = find_seed_dirs(args.data_dir)
    if not seed_dirs:
        raise SystemExit(f"no run_results.npy under {args.data_dir}")
    results = [np.load(os.path.join(d, "run_results.npy"), allow_pickle=False)
               for d in seed_dirs]

    problem = load_problem(seed_dirs)
    names = problem["names"]
    D = problem["num_vars"]
    n = len(results[0])
    second = (n % (2 * D + 2) == 0) and (n % (D + 2) != 0)
    params = {name: results[0][name] for name in names}

    for q in QOIS:
        stack = np.vstack([r[q] for r in results])
        Y = np.nanmean(stack, axis=0)
        res = sobol.analyze(problem, Y, calc_second_order=second, print_to_console=False)

        st_seeds = np.array([
            sobol.analyze(problem, stack[s], calc_second_order=second,
                          print_to_console=False)["ST"]
            for s in range(len(results))
        ])

        x = np.arange(D); w = 0.38
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, res["S1"], w, yerr=res["S1_conf"], capsize=3, label="S1", color="#79c")
        ax.bar(x + w / 2, res["ST"], w, yerr=res["ST_conf"], capsize=3, label="ST", color="#e89")
        for s in range(st_seeds.shape[0]):
            ax.scatter(x + w / 2, st_seeds[s], s=14, color="k", alpha=0.5, zorder=3)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("Sobol index")
        ax.set_title(f"{q}  (sum S1={res['S1'].sum():.2f})")
        ax.legend(title="bars: bootstrap CI | dots: per-seed ST")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"sobol_{q}.png"), dpi=140); plt.close(fig)

        fig, axs = plt.subplots(1, D, figsize=(4 * D, 3.6), sharey=True)
        for j, name in enumerate(names):
            xp = params[name]
            axs[j].scatter(xp, Y, s=6, alpha=0.25, color="#357")
            cx, cy = binned_mean(xp, Y)
            if cx.size:
                axs[j].plot(cx, cy, "-o", color="#e60", ms=3, lw=1.5)
            r2 = variance_explained(xp, Y)
            axs[j].set_xlabel(name)
            axs[j].set_title(f"$R^2$={r2:.2f}", fontsize=10)
        axs[0].set_ylabel(q)
        fig.suptitle(f"{q} vs parameters (seed-averaged; orange = binned mean, "
                     f"$R^2$ = variance explained by binned mean)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"scatter_{q}.png"), dpi=140); plt.close(fig)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()