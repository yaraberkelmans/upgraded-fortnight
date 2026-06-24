"""Compare the original riot_model against riot_model_refactor.

Runs both implementations on identical seeds/parameters and checks that:
  * integer/count datacollector columns are EXACTLY equal, and
  * float columns match within a tight tolerance,
then reports the wall-clock speedup. Run from the project root:

    python compare_refactor.py
"""

import time

import numpy as np

from riot_model.riot_model import (
    RiotModel as RiotModelOrig,
    SegregationParams as SegOrig,
    RiotParams as RiotOrig,
)
from riot_model_refactor.riot_model import (
    RiotModel as RiotModelFast,
    SegregationParams as SegFast,
    RiotParams as RiotFast,
)

# Columns that must match to the integer; everything else is a float average.
INT_COLUMNS = [
    "Happy", "Unhappy", "Home", "Away", "Moves", "Police",
    "Fighting fans", "Arrests this step", "Total arrests", "In warmup",
]


def run(model_cls, seg_cls, riot_cls, *, seed, torus, steps):
    seg = seg_cls(seed=seed, torus=torus, steps=steps)
    riot = riot_cls()
    model = model_cls(seg, riot)
    start = time.perf_counter()
    model.run_model()
    elapsed = time.perf_counter() - start
    return model.datacollector.get_model_vars_dataframe(), elapsed


def compare_frames(df_orig, df_fast):
    """Return (ok, list_of_problem_messages)."""
    problems = []
    if df_orig.shape != df_fast.shape:
        problems.append(
            f"shape mismatch: original {df_orig.shape} vs refactor {df_fast.shape}"
        )
        return False, problems
    if list(df_orig.columns) != list(df_fast.columns):
        problems.append("column set/order differs")
        return False, problems

    for col in df_orig.columns:
        a = df_orig[col].to_numpy()
        b = df_fast[col].to_numpy()
        if col in INT_COLUMNS:
            if not np.array_equal(a, b):
                idx = np.argmax(a != b)
                problems.append(
                    f"[{col}] integer mismatch at step {idx}: {a[idx]} vs {b[idx]}"
                )
        else:
            if not np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
                diff = np.nanmax(np.abs(a - b))
                problems.append(f"[{col}] float mismatch, max|Δ|={diff:.3e}")
    return (len(problems) == 0), problems


def main():
    seeds = [1, 7, 42]
    toruses = [True, False]
    steps = 150

    all_ok = True
    total_orig = 0.0
    total_fast = 0.0

    print(f"{'seed':>5} {'torus':>6} {'steps':>6} "
          f"{'orig (s)':>10} {'fast (s)':>10} {'speedup':>8}  result")
    print("-" * 64)

    for torus in toruses:
        for seed in seeds:
            df_o, t_o = run(RiotModelOrig, SegOrig, RiotOrig,
                            seed=seed, torus=torus, steps=steps)
            df_f, t_f = run(RiotModelFast, SegFast, RiotFast,
                            seed=seed, torus=torus, steps=steps)
            total_orig += t_o
            total_fast += t_f

            ok, problems = compare_frames(df_o, df_f)
            all_ok = all_ok and ok
            speedup = t_o / t_f if t_f > 0 else float("inf")
            result = "OK" if ok else "MISMATCH"
            print(f"{seed:>5} {str(torus):>6} {df_o.shape[0]:>6} "
                  f"{t_o:>10.3f} {t_f:>10.3f} {speedup:>7.2f}x  {result}")
            for p in problems:
                print(f"        - {p}")

    print("-" * 64)
    overall = total_orig / total_fast if total_fast > 0 else float("inf")
    print(f"TOTAL  original {total_orig:.3f}s  refactor {total_fast:.3f}s  "
          f"=> {overall:.2f}x overall")
    print("RESULT:", "ALL MATCH" if all_ok else "DIFFERENCES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
