"""Segregation-warmup feasibility sweep over the 3 parameters that drive it:
similarity_threshold, movement_decay, agent_density. Police are disabled so the
Schelling warmup is studied in isolation. Records how each run exits the warmup."""

import warnings
import numpy as np
from scipy.stats import qmc
from sklearn.tree import DecisionTreeClassifier, export_text

from riot_model import RiotModel, SegregationParams, RiotParams

N_SAMPLES = 400
WARMUP_BUDGET = 200      # max Schelling iterations before calling it stuck
GRID_N = 40
SEED = 42

RANGES = {
    "similarity_threshold": (0.00, 0.95),
    "movement_decay":       (0.05, 5.00),
    "agent_density":        (0.30, 0.95),
}


def classify_run(values):
    try:
        seg = SegregationParams(
            N=GRID_N, seed=SEED, steps=0,           # steps=0 -> skip init warmup
            similarity_threshold=float(values["similarity_threshold"]),
            movement_decay=float(values["movement_decay"]),
            agent_density=float(values["agent_density"]),
        )
        riot = RiotParams(police_density=0.0)
        model = RiotModel(segregation_params=seg, riot_params=riot)
    except Exception:
        return "construction_error"

    if not model.fans:
        return "construction_error"

    threshold = seg.warmup_entropy_threshold
    try:
        for _ in range(WARMUP_BUDGET):
            model.moves_this_step = 0
            for fan in model.fans:
                fan.move_if_unhappy()
            model.update_all_agents()
            if model.moves_this_step == 0:
                return "converged"
            if model.spatial_entropy() < threshold:
                return "segregated"
    except Exception:
        return "warmup_error"
    return "budget_exhausted"


def main():
    warnings.filterwarnings("ignore")

    names = list(RANGES.keys())
    lows = np.array([RANGES[n][0] for n in names])
    highs = np.array([RANGES[n][1] for n in names])

    sampler = qmc.LatinHypercube(d=len(names), optimization="random-cd", seed=SEED)
    X = qmc.scale(sampler.random(n=N_SAMPLES), lows, highs)

    labels = []
    for i, row in enumerate(X):
        labels.append(classify_run(dict(zip(names, row))))
        if (i + 1) % 50 == 0 or i == N_SAMPLES - 1:
            print(f"  {i + 1}/{N_SAMPLES}", flush=True)
    labels = np.array(labels)

    print("\nOutcomes:")
    for c, n in sorted(zip(*np.unique(labels, return_counts=True)),
                       key=lambda t: -t[1]):
        print(f"  {c:20s} {n:4d}  ({n / N_SAMPLES * 100:5.1f}%)")

    passed = np.isin(labels, ["converged", "segregated"]).astype(int)
    if 0 < passed.sum() < N_SAMPLES:
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)
        clf.fit(X, passed)
        print("\nRules for getting past warmup (class 1 = passed):")
        print(export_text(clf, feature_names=names, show_weights=True))
        Xp = X[passed == 1]
        print("Ranges among passed runs:")
        for j, n in enumerate(names):
            print(f"  {n:20s} [{Xp[:, j].min():.3f}, {Xp[:, j].max():.3f}]")
    else:
        print("\nAll runs in one class -- adjust ranges.")

    np.savez("warmup_raw.npz", X=X, labels=labels, names=names)


if __name__ == "__main__":
    main()