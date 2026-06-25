"""Feasibility sweep over 4 factors with deliberately wide ranges."""

import warnings
import numpy as np
from scipy.stats import qmc
from sklearn.tree import DecisionTreeClassifier, export_text

from riot_model import RiotModel, SegregationParams, RiotParams

N_SAMPLES = 400
WARMUP_CAP = 200
MAIN_STEPS = 40
GRID_N = 40
SEED = 42
ZERO_EPS = 0.01        # mean fight fraction below this -> degenerate zero
SAT_EPS = 0.99         # mean fight fraction above this -> degenerate saturated

# DELIBERATELY OVER-WIDE ranges -- we WANT to push into the broken corners.
#   similarity_threshold high -> Schelling never settles (warmup_no_converge)
#   fight_threshold high / cost high / police high -> fighting dies (zero)
#   fight_threshold low  / cost low                -> everyone fights (saturated)
#   police_density past grid capacity              -> construction error
RANGES = {
    "similarity_threshold": (0.00, 0.95),
    "fight_threshold":      (-0.40, 0.50),
    "hawk_dove_C":          (0.00, 20.0),
    "police_density":       (0.00, 0.15),
}

SEG_KEYS = {"similarity_threshold"}


def build_model(values):
    seg_kwargs = {"N": GRID_N, "seed": SEED}
    riot_kwargs = {}
    for name, v in values.items():
        if name in SEG_KEYS:
            seg_kwargs[name] = float(v)
        else:
            riot_kwargs[name] = float(v)
    return RiotModel(
        segregation_params=SegregationParams(**seg_kwargs),
        riot_params=RiotParams(**riot_kwargs),
    )


def classify_run(values):
    """Run one truncated simulation and return its failure class (a string)."""
    try:
        model = build_model(values)
    except Exception:
        return "construction_error"

    if len(model.fans) == 0:
        return "construction_error"

    try:
        for _ in range(WARMUP_CAP):
            if not model.in_warmup:
                break
            model.step()
        if model.in_warmup:
            return "warmup_no_converge"

        fight_fracs = []
        for _ in range(MAIN_STEPS):
            model.step()
            total = len(model.fans)
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

    sampler = qmc.LatinHypercube(d=len(names), optimization="random-cd", seed=SEED)
    X = qmc.scale(sampler.random(n=N_SAMPLES), lows, highs)

    print(f"Feasibility sweep: {N_SAMPLES} LHS points over wide ranges.\n")

    labels = []
    for i, row in enumerate(X):
        labels.append(classify_run(dict(zip(names, row))))
        if (i + 1) % 50 == 0 or i == N_SAMPLES - 1:
            print(f"  {i + 1}/{N_SAMPLES} points done", flush=True)
    labels = np.array(labels)

    print("\nOutcome breakdown:")
    classes, counts = np.unique(labels, return_counts=True)
    for c, n in sorted(zip(classes, counts), key=lambda t: -t[1]):
        print(f"  {c:22s} {n:4d}  ({n / N_SAMPLES * 100:5.1f}%)")

    valid = (labels == "valid").astype(int)
    if valid.sum() == 0 or valid.sum() == N_SAMPLES:
        print("\nAll points fell in one class -- widen or shift the ranges.")
        np.savez("feasibility_raw.npz", X=X, labels=labels, names=names)
        return

    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)
    clf.fit(X, valid)
    print("\nDecision-tree rules for VALID region (class 1 = valid):")
    print(export_text(clf, feature_names=names, show_weights=True))

    print("Empirical min/max of each parameter among VALID runs:")
    Xv = X[valid == 1]
    for j, n in enumerate(names):
        print(f"  {n:22s} [{Xv[:, j].min():.3f}, {Xv[:, j].max():.3f}]")

    np.savez("sa/feasibility_raw.npz", X=X, labels=labels, names=names)
    print("\nRaw results saved to feasibility_raw.npz")


if __name__ == "__main__":
    main()