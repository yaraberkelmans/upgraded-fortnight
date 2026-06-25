"""Local Sobol analysis from the combined Snellius outputs (cheap, run anywhere)."""

import numpy as np
from SALib.analyze import sobol

QOIS = [("Y_fight", "traj_fight"), ("Y_entropy", "traj_entropy")]


def burn_in_drift(traj, burn_in):
    tail = traj[:, burn_in:]
    half = tail.shape[1] // 2
    first = np.nanmean(tail[:, :half], axis=1)
    second = np.nanmean(tail[:, half:], axis=1)
    return float(np.nanmean(np.abs(second - first)))


def main():
    data = np.load("sobol_outputs.npz", allow_pickle=True)
    problem = {
        "num_vars": len(data["names"]),
        "names": list(data["names"]),
        "bounds": data["bounds"].tolist(),
    }
    calc_second_order = bool(data["calc_second_order"])
    names = problem["names"]
    burn_in = int(data["burn_in"])

    print(f"Saltelli rows: {data['X'].shape[0]} | sampled with SALib "
          f"{str(data['salib_version'])} | burn_in={burn_in}, tail={int(data['tail'])}")
    print(f"Mean warmup-converged fraction: {np.nanmean(data['converged_frac']):.2f} "
          f"(low -> indices partly reflect non-equilibrated Schelling states)")
    if np.nanmin(data["success_frac"]) < 1.0:
        print("WARNING: some rows had failed replicate seeds; check success_frac.")
    print()

    for qoi, traj_key in QOIS:
        Y = data[qoi]
        if np.isnan(Y).any():
            raise ValueError(f"{qoi} has missing rows; rerun combine after all tasks finish.")

        drift = burn_in_drift(data[traj_key], burn_in)
        res = sobol.analyze(problem, Y, calc_second_order=calc_second_order,
                            print_to_console=False)
        order = np.argsort(res["ST"])[::-1]

        print(f"=== {qoi} (ranked by ST) ===")
        print(f"Var(Y)={np.var(Y):.4g}  sum(S1)={res['S1'].sum():.3f}  "
              f"tail half-to-half drift={drift:.4g} (>> output spread suggests burn-in too short)")
        print(f"{'parameter':22s} {'S1':>8s} {'S1_conf':>8s} {'ST':>8s} {'ST_conf':>8s}")
        for j in order:
            print(f"{names[j]:22s} {res['S1'][j]:8.3f} {res['S1_conf'][j]:8.3f} "
                  f"{res['ST'][j]:8.3f} {res['ST_conf'][j]:8.3f}")
        if calc_second_order:
            print("second-order (S2):")
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    print(f"  {names[a]:20s} x {names[b]:20s} "
                          f"{res['S2'][a, b]:8.3f} +/- {res['S2_conf'][a, b]:.3f}")
        print()


if __name__ == "__main__":
    main()
