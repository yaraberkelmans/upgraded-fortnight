

import warnings
import numpy as np
from scipy.stats import qmc
from sklearn.tree import DecisionTreeClassifier, export_text

from riot_model import RiotModel, SegregationParams, RiotParams

# ---------------------------------------------------------------- knobs
N_SAMPLES = 400        # LHS points (a few hundred is plenty for a boundary)
WARMUP_STEPS = 30      # short Schelling warmup (set via SegregationParams.steps)
MAIN_STEPS = 30        # short violence phase -- enough to expose failures
GRID_N = 40            # lattice size; lower it to test the script cheaply
SEED = 42
COLLAPSE_FRAC = 0.5    # fan count below this fraction of initial -> collapse
ZERO_EPS = 0.01        # mean fight fraction below this -> degenerate zero
SAT_EPS = 0.99         # mean fight fraction above this -> degenerate saturated

# DELIBERATELY OVER-WIDE ranges -- we WANT to push into the broken corners.
RANGES = {
    "similarity_threshold": (0.00, 0.95),
    "perception_k":         (0.05, 3.00),
    "police_density":       (0.00, 0.25),
    "aggressiveness_mean":  (0.01, 0.99),
    "fight_threshold":      (-0.30, 0.60),
    "agent_density":        (0.30, 0.95),   # push toward grid saturation
    "movement_decay":       (0.05, 5.00),
    "logit_beta":           (0.10, 20.0),
}

# which dataclass each parameter belongs to
SEG_KEYS = {"similarity_threshold", "agent_density", "movement_decay"}


def build_model(values):
    """values: dict name->float. Routes into the two dataclasses and builds."""
    seg_kwargs = {"N": GRID_N, "seed": SEED, "steps": WARMUP_STEPS}
    riot_kwargs = {}
    for name, v in values.items():
        if name in SEG_KEYS:
            seg_kwargs[name] = float(v)
        else:
            riot_kwargs[name] = float(v)
    seg = SegregationParams(**seg_kwargs)
    riot = RiotParams(**riot_kwargs)
    return RiotModel(segregation_params=seg, riot_params=riot) 


def classify_run(values):
    """Run one truncated simulation and return its failure class (a string)."""
    try:
        model = build_model(values)
    except Exception:
        return "construction_error"

    n0 = len(model.fans)
    if n0 == 0:
        return "construction_error"

    fight_fracs = []
    try:
        for _ in range(MAIN_STEPS):
            model.step()
            total = len(model.fans)
            if total < COLLAPSE_FRAC * n0:
                return "fan_collapse"
            fight_fracs.append(model.count_fighting_fans() / total if total else 0.0)
    except Exception:
        return "runtime_error"

    mean_fight = float(np.mean(fight_fracs)) if fight_fracs else 0.0
    if mean_fight < ZERO_EPS:
        return "degenerate_zero"
    if mean_fight > SAT_EPS:
        return "degenerate_saturated"
    return "valid"


def main():
    warnings.filterwarnings("ignore")

    names = list(RANGES.keys())
    lows = np.array([RANGES[n][0] for n in names])
    highs = np.array([RANGES[n][1] for n in names])

    # scrambled / optimised LHS for better corner coverage in high-d
    sampler = qmc.LatinHypercube(d=len(names), optimization="random-cd", seed=SEED)
    unit = sampler.random(n=N_SAMPLES)
    X = qmc.scale(unit, lows, highs)

    print(f"Feasibility sweep: {N_SAMPLES} LHS points over wide ranges.\n")

    labels = []
    for i, row in enumerate(X):
        labels.append(classify_run(dict(zip(names, row))))
        if (i + 1) % 50 == 0 or i == N_SAMPLES - 1:
            print(f"  {i + 1}/{N_SAMPLES} points done", flush=True)
    labels = np.array(labels)

    # ---- breakdown by failure class ----
    print("\nOutcome breakdown:")
    classes, counts = np.unique(labels, return_counts=True)
    for c, n in sorted(zip(classes, counts), key=lambda t: -t[1]):
        print(f"  {c:22s} {n:4d}  ({n / N_SAMPLES * 100:5.1f}%)")

    valid = (labels == "valid").astype(int)
    if valid.sum() == 0 or valid.sum() == N_SAMPLES:
        print("\nAll points fell in one class -- widen or shift the ranges.")
        np.savez("feasibility_raw.npz", X=X, labels=labels, names=names)
        return

    # ---- decision-tree boundary -> readable rules for clipping ----
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)
    clf.fit(X, valid)
    print("\nDecision-tree rules for VALID region (class 1 = valid):")
    print(export_text(clf, feature_names=names, show_weights=True))

    # ---- empirical valid-only ranges (a quick clip suggestion) ----
    print("Empirical min/max of each parameter among VALID runs")
    print("(use as a first-cut clip; combine with the tree rules above):")
    Xv = X[valid == 1]
    for j, n in enumerate(names):
        print(f"  {n:22s} [{Xv[:, j].min():.3f}, {Xv[:, j].max():.3f}]")

    np.savez("sweeps/feasibility_raw.npz", X=X, labels=labels, names=names)
    print("\nRaw results saved to feasibility_raw.npz")


if __name__ == "__main__":
    main()