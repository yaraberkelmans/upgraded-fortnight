
import warnings
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

from riot_model import RiotModel, SegregationParams, RiotParams


N = 512            # Saltelli base sample -> N*(d+2) = N*6 model runs
BURN_IN = 100      # main-loop steps discarded before averaging
TAIL = 100         # main-loop steps averaged into the Sobol outputs
CV_WINDOW = 25     # window length (steps) for the CV diagnostic
CV_TOL = 0.05      # |CV_last - CV_prev| below this == "stabilised"
ZERO_EPS = 1e-6    # below this mean, CV is undefined (no-violence regime)
GRID_N = 40        
SEED = 42          

problem = {
    "num_vars": 4,
    "names": ["similarity_threshold", "perception_k",
              "police_density", "aggressiveness_mean"],
    "bounds": [[0.10, 0.70],   # similarity_threshold
               [0.20, 1.50],   # perception_k
               [0.01, 0.15],   # police_density
               [0.05, 0.95]],  # aggressiveness_mean
}


def windowed_cv(series, window):
    """CV (std/mean) per consecutive non-overlapping window.

    Returns the list of window CVs, or NaN for windows whose mean is ~0
    (CV is meaningless there). Uses the trailing windows only.
    """
    series = np.asarray(series, dtype=float)
    n_full = len(series) // window
    cvs = []
    for w in range(n_full):
        chunk = series[w * window:(w + 1) * window]
        mean = chunk.mean()
        if abs(mean) < ZERO_EPS:
            cvs.append(np.nan)
        else:
            cvs.append(chunk.std() / mean)
    return cvs


def has_stabilised(series, window, tol):
    """True if the last two window-CVs differ by less than tol.

    Returns (stabilised, degenerate). 'degenerate' flags the near-zero-mean
    case where CV can't be evaluated (e.g. fighting fraction stuck at ~0).
    """
    cvs = windowed_cv(series, window)
    if len(cvs) < 2:
        return False, False
    last, prev = cvs[-1], cvs[-2]
    if np.isnan(last) or np.isnan(prev):
        return False, True
    return abs(last - prev) < tol, False


def run_one(x):
    """One model evaluation for a parameter row x.

    Returns mean entropy, mean fighting fraction (both over the tail), and the
    CV diagnostics for entropy and fighting fraction.
    """
    st, pk, pd, am = x

    seg = SegregationParams(N=GRID_N, seed=SEED, similarity_threshold=float(st))
    riot = RiotParams(perception_k=float(pk),
                      police_density=float(pd),
                      aggressiveness_mean=float(am))
    # RiotModel.__init__ already runs the Schelling warmup. The loop below is
    # the violence phase settling into dynamic equilibrium, then the tail.
    model = RiotModel(segregation_params=seg, riot_params=riot)

    ent_series, fight_series = [], []
    for _ in range(BURN_IN + TAIL):
        model.step()
        total = len(model.fans)
        ent_series.append(model.spatial_entropy())
        fight_series.append(model.count_fighting_fans() / total if total else 0.0)

    ent_series = np.array(ent_series)
    fight_series = np.array(fight_series)

    mean_ent = ent_series[-TAIL:].mean()
    mean_fight = fight_series[-TAIL:].mean()

    ent_stable, _ = has_stabilised(ent_series, CV_WINDOW, CV_TOL)
    fight_stable, fight_degen = has_stabilised(fight_series, CV_WINDOW, CV_TOL)

    return mean_ent, mean_fight, ent_stable, fight_stable, fight_degen


def main():
    warnings.filterwarnings("ignore") 

    X = saltelli.sample(problem, N, calc_second_order=False)
    n_runs = X.shape[0]
    print(f"Sobol SA: {n_runs} model runs (N={N}, d={problem['num_vars']}, "
          f"cost N*(d+2)).")

    Y_ent = np.empty(n_runs)
    Y_fight = np.empty(n_runs)
    ent_stab = np.zeros(n_runs, dtype=bool)
    fight_stab = np.zeros(n_runs, dtype=bool)
    fight_degen = np.zeros(n_runs, dtype=bool)

    for i, x in enumerate(X):
        Y_ent[i], Y_fight[i], ent_stab[i], fight_stab[i], fight_degen[i] = run_one(x)
        if (i + 1) % 50 == 0 or i == n_runs - 1:
            print(f"  {i + 1}/{n_runs} runs done", flush=True)

    #
    print("\nEquilibrium diagnostics (CV stabilisation over the run):")
    print(f"spatial entropy stabilised: {ent_stab.mean() * 100:5.1f}% of runs"
          f"(this is the equilibrium clock)")
    print(f"fighting fraction stabilised: {fight_stab.mean() * 100:5.1f}% of runs")
    print(f"fighting fraction ~0 (no-violence regime): "
          f"{fight_degen.mean() * 100:5.1f}% of runs")
    if ent_stab.mean() < 0.9:
        print("NOTE: <90% of runs reached entropy equilibrium -- consider a "
              "longer BURN_IN/TAIL.")

   
    np.savez("data/sa/sa_raw_outputs.npz", X=X, Y_entropy=Y_ent, Y_fight=Y_fight,
             ent_stable=ent_stab, fight_stable=fight_stab, fight_degen=fight_degen)

    for label, Y in [("SPATIAL ENTROPY", Y_ent), ("FIGHTING FRACTION", Y_fight)]:
        Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
        print(f"\n=== {label} ===")
        print(f"{'parameter':22s} {'S1':>8s} {'S1_conf':>8s} {'ST':>8s} {'ST_conf':>8s}")
        for name, s1, s1c, st, stc in zip(problem["names"], Si["S1"],
                                           Si["S1_conf"], Si["ST"], Si["ST_conf"]):
            print(f"{name:22s} {s1:8.3f} {s1c:8.3f} {st:8.3f} {stc:8.3f}")

    print("\nRaw outputs saved to sa_raw_outputs.npz")


if __name__ == "__main__":
    main()