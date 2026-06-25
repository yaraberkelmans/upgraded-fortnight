"""Snellius worker: evaluate a contiguous chunk of Saltelli rows.

Protocol per (row, seed): Schelling warmup (capped, NOT recorded) -> fixed
VIOLENCE_BURN_IN steps -> TAIL steps. The burn-in + tail are logged step by step
into a structured array, so any quantity of interest (and the question of whether
the burn-in was long enough) can be re-derived at analysis time without rerunning.
Replicate seeds use common random numbers and are kept separate (not pre-averaged).

Per-row output -> runs/row_<i>.npz; combine.py reduces them in index order.
"""

import os
import sys
import numpy as np

from riot_model import RiotModel, SegregationParams, RiotParams, FanGroup

WARMUP_CAP = 200
VIOLENCE_BURN_IN = 100
TAIL = 100
RECORD = VIOLENCE_BURN_IN + TAIL
GRID_N = 40
REPLICATE_SEEDS = [10, 20, 30, 40, 50]
OUT_DIR = "runs"

SEG_KEYS = {"similarity_threshold"}

STATE_DTYPE = np.dtype([
    ("step", np.int32),
    ("in_warmup", np.bool_),
    ("n_fans", np.int32),
    ("home", np.int32),
    ("away", np.int32),
    ("police", np.int32),
    ("happy", np.int32),
    ("unhappy", np.int32),
    ("fighting", np.int32),
    ("moves", np.int32),
    ("arrests_step", np.int32),
    ("total_arrests", np.int32),
    ("avg_similarity", np.float64),
    ("spatial_entropy", np.float64),
    ("spatial_entropy_local", np.float64),
    ("entropy_cv_local", np.float64),
    ("avg_aggressiveness", np.float64),
    ("avg_win_probability", np.float64),
    ("avg_arrest_probability", np.float64),
    ("avg_move_distance", np.float64),
    ("avg_move_distance_moved", np.float64),
    ("fighting_fraction", np.float64),
    ("home_fraction_current", np.float64),
])


def record_state(rec, step, model):
    n = len(model.fans)
    home = model.count_group(FanGroup.HOME)
    fighting = model.count_fighting_fans()
    rec["step"] = step
    rec["in_warmup"] = model.in_warmup
    rec["n_fans"] = n
    rec["home"] = home
    rec["away"] = model.count_group(FanGroup.AWAY)
    rec["police"] = model.count_police()
    rec["happy"] = model.count_happy()
    rec["unhappy"] = model.count_unhappy()
    rec["fighting"] = fighting
    rec["moves"] = model.moves_this_step
    rec["arrests_step"] = model.arrests_this_step
    rec["total_arrests"] = model.total_arrests
    rec["avg_similarity"] = model.average_similarity()
    rec["spatial_entropy"] = model._zone_entropy(model.segregation_params.zone_size)
    rec["spatial_entropy_local"] = model.spatial_entropy_fine()
    rec["entropy_cv_local"] = model.entropy_cv_fine()
    rec["avg_aggressiveness"] = model.average_aggressiveness()
    rec["avg_win_probability"] = model.average_perceived_win_probability()
    rec["avg_arrest_probability"] = model.average_perceived_arrest_probability()
    rec["avg_move_distance"] = model.average_last_move_distance()
    rec["avg_move_distance_moved"] = model.average_last_move_distance_of_moved_fans()
    rec["fighting_fraction"] = fighting / n if n else 0.0
    rec["home_fraction_current"] = home / n if n else 0.0


def run_once(values, seed):
    states = np.zeros(RECORD, dtype=STATE_DTYPE)
    seg_kwargs = {"N": GRID_N, "seed": seed}
    riot_kwargs = {}
    for name, v in values.items():
        if name in SEG_KEYS:
            seg_kwargs[name] = float(v)
        else:
            riot_kwargs[name] = float(v)

    try:
        model = RiotModel(
            segregation_params=SegregationParams(**seg_kwargs),
            riot_params=RiotParams(**riot_kwargs),
        )
        if len(model.fans) == 0:
            return states, 0, 0, 0

        warmup_steps = 0
        converged = 0
        for _ in range(WARMUP_CAP):
            if not model.in_warmup:
                converged = 1
                break
            model.step()
            warmup_steps += 1
        model.in_warmup = False

        for s in range(RECORD):
            model.step()
            record_state(states[s], s, model)
        return states, warmup_steps, converged, 1
    except Exception:
        return states, 0, 0, 0


def run_row(values):
    r = len(REPLICATE_SEEDS)
    states = np.zeros((r, RECORD), dtype=STATE_DTYPE)
    warmup_steps = np.zeros(r, np.int32)
    converged = np.zeros(r, np.int32)
    success = np.zeros(r, np.int32)
    for k, seed in enumerate(REPLICATE_SEEDS):
        states[k], warmup_steps[k], converged[k], success[k] = run_once(values, seed)
    return states, warmup_steps, converged, success


def main():
    start = int(sys.argv[1])
    count = int(sys.argv[2])

    data = np.load("sobol_sample.npz", allow_pickle=True)
    X = data["X"]
    names = list(data["names"])

    os.makedirs(OUT_DIR, exist_ok=True)
    end = min(start + count, X.shape[0])

    for i in range(start, end):
        path = os.path.join(OUT_DIR, f"row_{i:06d}.npz")
        if os.path.exists(path):
            continue
        states, warmup_steps, converged, success = run_row(dict(zip(names, X[i])))
        np.savez(
            path,
            index=i,
            states=states,
            seeds=np.array(REPLICATE_SEEDS),
            warmup_steps=warmup_steps,
            converged=converged,
            success=success,
            burn_in=VIOLENCE_BURN_IN,
            tail=TAIL,
        )
        if (i - start + 1) % 20 == 0 or i == end - 1:
            print(f"  rows {start}..{i} done", flush=True)


if __name__ == "__main__":
    main()
