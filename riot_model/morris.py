

import warnings
import numpy as np
from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze

from riot_model import RiotModel, SegregationParams, RiotParams

# ---------------------------------------------------------------- knobs
R_TRAJECTORIES = 10    # Morris trajectories; cost = R*(k+1) runs
N_LEVELS = 4           # grid levels per factor (Morris standard)
BURN_IN = 100          # main-loop steps discarded before averaging
TAIL = 100             # main-loop steps averaged into the output
GRID_N = 40            # lattice size; lower it to test the script cheaply
SEED = 42

problem = {
    "num_vars": 7,
    "names": ["similarity_threshold", "perception_k", "police_density",
              "fight_threshold", "agent_density",
              "movement_decay", "logit_beta"],
    "bounds": [[0.10, 0.70],    # similarity_threshold
               [0.20, 1.50],    # perception_k
               [0.01, 0.15],    # police_density   
               [-0.10, 0.30],   # fight_threshold
               [0.50, 0.85],    # agent_density
               [0.20, 3.00],    # movement_decay
               [1.00, 10.0]],   # logit_beta
}

# which dataclass each parameter belongs to
SEG_KEYS = {"similarity_threshold", "agent_density", "movement_decay"}


def run_one(x):
    """One model evaluation -> (mean entropy, mean fighting fraction) over tail."""
    seg_kwargs = {"N": GRID_N, "seed": SEED}
    riot_kwargs = {}
    for name, v in zip(problem["names"], x):
        if name in SEG_KEYS:
            seg_kwargs[name] = float(v)
        else:
            riot_kwargs[name] = float(v)

    seg = SegregationParams(**seg_kwargs)
    riot = RiotParams(**riot_kwargs)
    model = RiotModel(segregation_params=seg, riot_params=riot)  # runs warmup

    ent, fight = [], []
    for _ in range(BURN_IN + TAIL):
        model.step()
        total = len(model.fans)
        ent.append(model.spatial_entropy())
        fight.append(model.count_fighting_fans() / total if total else 0.0)

    return float(np.mean(ent[-TAIL:])), float(np.mean(fight[-TAIL:]))


def main():
    warnings.filterwarnings("ignore")

    X = morris_sample(problem, N=R_TRAJECTORIES, num_levels=N_LEVELS)
    n_runs = X.shape[0]
    print(f"Morris screen: {n_runs} model runs "
          f"(R={R_TRAJECTORIES}, k={problem['num_vars']}, cost R*(k+1)).\n")

    Y_ent = np.empty(n_runs)
    Y_fight = np.empty(n_runs)
    for i, x in enumerate(X):
        Y_ent[i], Y_fight[i] = run_one(x)
        if (i + 1) % 20 == 0 or i == n_runs - 1:
            print(f"  {i + 1}/{n_runs} runs done", flush=True)

    for label, Y in [("SPATIAL ENTROPY", Y_ent), ("FIGHTING FRACTION", Y_fight)]:
        res = morris_analyze(problem, X, Y, num_levels=N_LEVELS,
                             print_to_console=False)
        order = np.argsort(res["mu_star"])[::-1]  # most influential first
        print(f"\n=== {label} (ranked by mu_star) ===")
        print(f"{'parameter':22s} {'mu_star':>9s} {'mu':>9s} {'sigma':>9s}")
        for j in order:
            print(f"{res['names'][j]:22s} {res['mu_star'][j]:9.4f} "
                  f"{res['mu'][j]:9.4f} {res['sigma'][j]:9.4f}")

    np.savez("morris_raw_outputs.npz", X=X, Y_entropy=Y_ent, Y_fight=Y_fight)
    print("\nmu_star = overall influence (rank by this).")
    print("sigma   = high relative to mu_star means non-linear / interacting.")
    print("Raw outputs saved to morris_raw_outputs.npz")


if __name__ == "__main__":
    main()