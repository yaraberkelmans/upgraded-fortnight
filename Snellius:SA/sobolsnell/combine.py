"""Reduce per-row step logs into one analysis file (run on Snellius, ship result).

For each row it takes the TAIL slice of the recorded window, computes the tail
mean per replicate seed, then averages over the seeds that ran. It keeps the
per-seed spread and the seed-averaged tail trajectories so burn-in adequacy and
stationarity can be checked at home without the full ~GB of step logs.
"""

import os
import glob
import numpy as np

OUT_DIR = "runs"


def main():
    sample = np.load("sobol_sample.npz", allow_pickle=True)
    X = sample["X"]
    n = X.shape[0]

    files = glob.glob(os.path.join(OUT_DIR, "row_*.npz"))
    if not files:
        raise SystemExit("no row files found in runs/")

    probe = np.load(files[0])
    r = probe["seeds"].shape[0]
    burn_in = int(probe["burn_in"])
    tail = int(probe["tail"])
    record = burn_in + tail

    Y_fight = np.full(n, np.nan)
    Y_entropy = np.full(n, np.nan)
    fight_per_seed = np.full((n, r), np.nan)
    entropy_per_seed = np.full((n, r), np.nan)
    converged_frac = np.full(n, np.nan)
    success_frac = np.full(n, np.nan)
    warmup_steps_mean = np.full(n, np.nan)
    traj_fight = np.full((n, record), np.nan)
    traj_entropy = np.full((n, record), np.nan)

    for f in files:
        d = np.load(f)
        i = int(d["index"])
        states = d["states"]                 # (r, record) structured
        ok = d["success"].astype(bool)

        fseed = states["fighting_fraction"][:, burn_in:].mean(axis=1)
        eseed = states["spatial_entropy_local"][:, burn_in:].mean(axis=1)
        fight_per_seed[i] = fseed
        entropy_per_seed[i] = eseed

        if ok.any():
            Y_fight[i] = fseed[ok].mean()
            Y_entropy[i] = eseed[ok].mean()
            traj_fight[i] = states["fighting_fraction"][ok].mean(axis=0)
            traj_entropy[i] = states["spatial_entropy_local"][ok].mean(axis=0)

        converged_frac[i] = d["converged"].mean()
        success_frac[i] = ok.mean()
        warmup_steps_mean[i] = d["warmup_steps"].mean()

    missing = int(np.isnan(Y_fight).sum())
    if missing:
        print(f"WARNING: {missing} rows missing/all-failed -- rerun those tasks "
              f"before analysing (Sobol needs every row).")

    np.savez(
        "sobol_outputs.npz",
        X=X,
        names=sample["names"],
        bounds=sample["bounds"],
        N=sample["N"],
        calc_second_order=sample["calc_second_order"],
        sample_seed=sample["sample_seed"],
        salib_version=sample["salib_version"],
        burn_in=burn_in,
        tail=tail,
        Y_fight=Y_fight,
        Y_entropy=Y_entropy,
        fight_per_seed=fight_per_seed,
        entropy_per_seed=entropy_per_seed,
        converged_frac=converged_frac,
        success_frac=success_frac,
        warmup_steps_mean=warmup_steps_mean,
        traj_fight=traj_fight,
        traj_entropy=traj_entropy,
    )
    print(f"Combined {n - missing}/{n} rows into sobol_outputs.npz")


if __name__ == "__main__":
    main()
