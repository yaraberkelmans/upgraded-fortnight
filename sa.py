"""Sensitivity analysis for the Mixed Football Riots model.

Output metric: total arrests per run.
Layers: A = OFAT, B = Morris screening, C = Sobol.

Place next to mixed_models.py, fan.py, cop.py. Needs SALib for Morris/Sobol.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mixed_models import MixedFootballRiotsModel, MixedFootballRiotsParams


# --- run settings ---
GRID_N = 30
N_STEPS = 100
REPLICATIONS = 3
BASE_SEED = 1000

RUN_OFAT = True
RUN_MORRIS = True
RUN_SOBOL = True

OFAT_POINTS = 7
MORRIS_TRAJECTORIES = 8
MORRIS_LEVELS = 4
SOBOL_N = 64               # power of 2; Sobol runs = SOBOL_N * (D + 2)
SOBOL_SECOND_ORDER = False

OUTPUT_DIR = "sa_output"


# --- parameters and bounds (from the server_mix sliders) ---
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "fan_density":            (0.30, 0.95),
    "cop_density":            (0.01, 0.25),
    "home_fraction":          (0.20, 0.80),
    "aggression_threshold":   (-1.0, 1.0),
    "base_aggression_weight": (0.0, 3.0),
    "rival_weight":           (0.0, 3.0),
    "own_group_weight":       (0.0, 3.0),
    "riot_contagion_weight":  (0.0, 3.0),
    "police_deterrence":      (0.0, 3.0),
    "risk_weight":            (0.0, 3.0),
}


def total_arrests(overrides: dict, seed: int) -> float:
    kwargs = dict(overrides)
    kwargs["seed"] = seed
    kwargs["N"] = GRID_N
    kwargs["steps"] = N_STEPS
    model = MixedFootballRiotsModel(**kwargs)
    model.run_model()
    df = model.datacollector.get_model_vars_dataframe()
    return float(df["Arrests"].sum())


def mean_arrests(overrides):
    values = [total_arrests(overrides, BASE_SEED + r) for r in range(REPLICATIONS)]
    return float(np.mean(values)), float(np.std(values))


def baseline_value(name):
    return getattr(MixedFootballRiotsParams(), name)


def problem_def():
    names = list(PARAM_BOUNDS.keys())
    return {
        "num_vars": len(names),
        "names": names,
        "bounds": [list(PARAM_BOUNDS[n]) for n in names],
    }


# --- Layer A: OFAT ---
def run_ofat():
    print("\nOFAT")
    rows = []
    for i, (name, (low, high)) in enumerate(PARAM_BOUNDS.items(), start=1):
        print(f"[{i}/{len(PARAM_BOUNDS)}] {name}")
        for value in np.linspace(low, high, OFAT_POINTS):
            mean, std = mean_arrests({name: float(value)})
            rows.append({
                "parameter": name,
                "value": float(value),
                "baseline": baseline_value(name),
                "mean_arrests": mean,
                "std_arrests": std,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "ofat_results.csv"), index=False)
    plot_ofat(df)
    return df


def plot_ofat(df):
    params = list(PARAM_BOUNDS.keys())
    cols = 3
    rows = int(np.ceil(len(params) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, params):
        sub = df[df["parameter"] == name].sort_values("value")
        ax.errorbar(sub["value"], sub["mean_arrests"], yerr=sub["std_arrests"],
                    marker="o", capsize=3)
        ax.axvline(baseline_value(name), color="grey", ls="--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("total arrests")

    for ax in axes[len(params):]:
        ax.axis("off")

    fig.suptitle("OFAT: total arrests vs each parameter")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ofat_plot.png"), dpi=130)
    plt.close(fig)



def run_morris():
    print("\nMorris")
    try:
        from SALib.sample import morris as morris_sample
        from SALib.analyze import morris as morris_analyze
    except ImportError:
        print("SALib not installed; skipping. pip install SALib")
        return pd.DataFrame()

    problem = problem_def()
    X = morris_sample.sample(problem, N=MORRIS_TRAJECTORIES, num_levels=MORRIS_LEVELS)
    print(f"{len(X)} sets x {REPLICATIONS} reps")

    Y = np.empty(len(X))
    for j, row in enumerate(X):
        Y[j], _ = mean_arrests({n: float(v) for n, v in zip(problem["names"], row)})
        if (j + 1) % 5 == 0 or j == len(X) - 1:
            print(f"  {j + 1}/{len(X)}")

    Si = morris_analyze.analyze(problem, X, Y, num_levels=MORRIS_LEVELS,
                                print_to_console=False)
    df = pd.DataFrame({
        "parameter": problem["names"],
        "mu_star": Si["mu_star"],
        "mu_star_conf": Si["mu_star_conf"],
        "sigma": Si["sigma"],
        "mu": Si["mu"],
    }).sort_values("mu_star", ascending=False)

    df.to_csv(os.path.join(OUTPUT_DIR, "morris_results.csv"), index=False)
    plot_morris(df)
    return df


def plot_morris(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.barh(df["parameter"], df["mu_star"], xerr=df["mu_star_conf"], capsize=3)
    ax1.invert_yaxis()
    ax1.set_xlabel("mu*")
    ax1.set_title("Morris: influence")

    ax2.scatter(df["mu_star"], df["sigma"])
    for _, r in df.iterrows():
        ax2.annotate(r["parameter"], (r["mu_star"], r["sigma"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("mu*")
    ax2.set_ylabel("sigma")
    ax2.set_title("Morris: mu* vs sigma")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "morris_plot.png"), dpi=130)
    plt.close(fig)



def run_sobol():
    print("\nSobol")
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
    

    problem = problem_def()
    X = sobol_sample.sample(problem, SOBOL_N,
                            calc_second_order=SOBOL_SECOND_ORDER)
    print(f"{len(X)} sets x {REPLICATIONS} reps")

    Y = np.empty(len(X))
    for j, row in enumerate(X):
        Y[j], _ = mean_arrests({n: float(v) for n, v in zip(problem["names"], row)})
        if (j + 1) % 25 == 0 or j == len(X) - 1:
            print(f"  {j + 1}/{len(X)}")

    Si = sobol_analyze.analyze(problem, Y,
                               calc_second_order=SOBOL_SECOND_ORDER,
                               print_to_console=False)
    df = pd.DataFrame({
        "parameter": problem["names"],
        "S1": Si["S1"],
        "S1_conf": Si["S1_conf"],
        "ST": Si["ST"],
        "ST_conf": Si["ST_conf"],
    }).sort_values("ST", ascending=False)

    df.to_csv(os.path.join(OUTPUT_DIR, "sobol_results.csv"), index=False)

    if SOBOL_SECOND_ORDER:
        names = problem["names"]
        s2_rows = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                s2_rows.append({
                    "pair": f"{names[a]} x {names[b]}",
                    "S2": Si["S2"][a, b],
                    "S2_conf": Si["S2_conf"][a, b],
                })
        pd.DataFrame(s2_rows).to_csv(
            os.path.join(OUTPUT_DIR, "sobol_second_order.csv"), index=False)

    plot_sobol(df)
    return df


def plot_sobol(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))
    h = 0.4
    ax.barh(y - h / 2, df["S1"], height=h, xerr=df["S1_conf"], capsize=3,
            label="S1 (first order)")
    ax.barh(y + h / 2, df["ST"], height=h, xerr=df["ST_conf"], capsize=3,
            label="ST (total)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index")
    ax.set_title("Sobol: first-order vs total-order")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "sobol_plot.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = time.time()
    print(f"total arrests | grid={GRID_N} steps={N_STEPS} reps={REPLICATIONS} "
          f"params={len(PARAM_BOUNDS)}")

    if RUN_OFAT:
        run_ofat()
    if RUN_MORRIS:
        run_morris()
    if RUN_SOBOL:
        run_sobol()

    print(f"\ndone in {time.time() - start:.0f}s -> ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()