"""Generate the Saltelli sample once. Runs + analysis both load this file,
so indices stay consistent (same problem, N, second-order flag, SALib version)."""

import numpy as np
import SALib
from importlib.metadata import version

try:
    from SALib.sample.sobol import sample as saltelli_sample
except ImportError:
    from SALib.sample.saltelli import sample as saltelli_sample

N = 512
CALC_SECOND_ORDER = True
SAMPLE_SEED = 42


RANGES = {
    "similarity_threshold": (0.1, 0.8),
    "fight_threshold":      (-0.40, 0.41),
    "hawk_dove_C":          (0.18, 19.85),
    "police_density":       (0.00, 0.18),
}

problem = {
    "num_vars": len(RANGES),
    "names": list(RANGES.keys()),
    "bounds": [list(b) for b in RANGES.values()],
}


def main():
    X = saltelli_sample(problem, N, calc_second_order=CALC_SECOND_ORDER,
                        seed=SAMPLE_SEED)
    
    salib_ver = version("SALib")
    
    np.savez(
        "sobol_sample.npz",
        X=X,
        names=np.array(problem["names"]),
        bounds=np.array(problem["bounds"]),
        N=N,
        calc_second_order=CALC_SECOND_ORDER,
        sample_seed=SAMPLE_SEED,
        salib_version=salib_ver,
    )
    print(f"Saltelli sample: {X.shape[0]} rows x {X.shape[1]} params "
          f"(N={N}, second_order={CALC_SECOND_ORDER}, SALib.")
    print("Saved to sobol_sample.npz")


if __name__ == "__main__":
    main()