"""Grid sweep of home_fraction x agent_density on mean fighting fraction.

Mirrors run_sobol.py's per-run structure (spatial warmup -> riot burn-in ->
100-step measurement) but sweeps a 2D parameter grid instead of a Sobol sample.
One summary row per (home_fraction, agent_density) cell. Run once per seed; the
seeds are replicates, handled by the outer batch loop exactly like the Sobol job.

Output
------
- ``<output-dir>/grid_results.npy``: one RESULT_DTYPE row per grid cell.
- ``<output-dir>/metadata.json``: grid axes + fixed parameters.

Run from project root:
    python EXPERIMENTS/ofat/ofat.py --seed 43 --workers $N --output-dir data_ofat/seed_43
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from model.riot_model import RiotModel, RiotParams, SegregationParams

# ── Swept axes ────────────────────────────────────────────────────────────────────────────────

# home_fraction spans 0.1–0.9 so the surface can be checked for symmetry about
# 0.5 (home and away are interchangeable, and the aggressiveness coupling swaps
# with them). Trim to arange(0.50, 0.901, 0.05) if you only want one half.
HOME_FRACTIONS = np.round(np.arange(0.10, 0.901, 0.10), 4)   # 9 points
AGENT_DENSITIES = np.round(np.arange(0.30, 0.851, 0.05), 4)  # 12 points

# ── Fixed parameters (Sobol-swept values pinned here) ────────────────────────────────────────

FIXED = {
    "similarity_threshold": 0.30,  # low enough to converge at high density
    "fight_threshold": 0.0,
    "hawk_dove_C": 4.0,
    "police_density": 0.05,
}

# ── Experiment constants ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_SEED = 43
RIOT_BURN_IN_STEPS = 100 # steps discarded after spatial warmup before measurement begins
MEASUREMENT_STEPS = 100 # steps over which all QoIs are averaged
MAX_SPATIAL_WARMUP_STEPS = 2_000

# ── Numpy record dtype ────────────────────────────────────────────────────────────────────────

RESULT_DTYPE = np.dtype([
    ("home_fraction", np.float64),
    ("agent_density", np.float64),
    ("seed", np.int64),
    ("valid", np.bool_),
    ("failure_code", "U64"),
    ("warmup_steps", np.int32),
    ("runtime_seconds", np.float64),
    ("mean_fighting_fraction", np.float64),
    ("mean_fighting", np.float64),
    ("peak_fighting", np.float64),
    ("mean_arrests_per_step", np.float64),
    ("mean_spatial_entropy_local", np.float64),
    ("mean_happy_fraction", np.float64),
])

# ── Helper functions ──────────────────────────────────────────────────────────────────────────


def _fine_entropy(model: RiotModel) -> float:
    return float(model._zone_entropy(model.segregation_params.zone_size_fine))


def _failed(home_fraction, agent_density, seed, warmup_steps, started, code):
    nan = float("nan")
    return (
        float(home_fraction), float(agent_density), int(seed),
        False, code, int(warmup_steps), time.perf_counter() - started,
        nan, nan, nan, nan, nan, nan,
    )


# ── Worker entry-point ────────────────────────────────────────────────────────────────────────


def run_one(job: tuple[float, float, int]) -> tuple:
    home_fraction, agent_density, seed = job
    started = time.perf_counter()

    segregation = SegregationParams(
        N=40,
        agent_density=float(agent_density),
        home_fraction=float(home_fraction),
        similarity_threshold=FIXED["similarity_threshold"],
        movement_decay=1.0,
        seed=int(seed),
        torus=True,
        count_empty_as_different=True,
        zone_size=10,
        zone_size_fine=4,
        warmup_cv_threshold=0.025,
        warmup_window=10,
        random_move_chance=0.005,
        collect_data=False,
    )
    riot = RiotParams(
        police_density=FIXED["police_density"],
        perception_k=0.693,
        fan_vision=2,
        fight_threshold=FIXED["fight_threshold"],
        police_vision=5,
        logit_beta=5.0,
        hawk_dove_strategy="logit_prior",
        hawk_dove_C=FIXED["hawk_dove_C"],
        aggressiveness_mean=None,            # native home_fraction coupling kept
        aggressiveness_concentration=12.0,
        fighting_enabled=True,
    )

    try:
        model = RiotModel(segregation_params=segregation, riot_params=riot)
    except Exception as exc:
        return _failed(home_fraction, agent_density, seed, 0, started,
                       f"init:{type(exc).__name__}")

    warmup_steps = 0
    try:
        while model.in_warmup and warmup_steps < MAX_SPATIAL_WARMUP_STEPS:
            model.step()
            warmup_steps += 1
    except Exception as exc:
        return _failed(home_fraction, agent_density, seed, warmup_steps, started,
                       f"warmup:{type(exc).__name__}")

    if model.in_warmup:
        return _failed(home_fraction, agent_density, seed, warmup_steps, started,
                       "warmup_not_converged")

    try:
        for _ in range(RIOT_BURN_IN_STEPS):
            model.step()

        # Population is conserved (every arrest respawns a fan), so n_fans is
        # constant across the measurement window.
        n_fans = max(len(model.fans), 1)
        fighting = np.empty(MEASUREMENT_STEPS)
        arrests = np.empty(MEASUREMENT_STEPS)
        entropy = np.empty(MEASUREMENT_STEPS)
        happy = np.empty(MEASUREMENT_STEPS)

        for t in range(MEASUREMENT_STEPS):
            model.step()
            fighting[t] = model.count_fighting_fans()
            arrests[t] = model.arrests_this_step
            entropy[t] = _fine_entropy(model)
            happy[t] = model.count_happy()
    except Exception as exc:
        return _failed(home_fraction, agent_density, seed, warmup_steps, started,
                       f"riot:{type(exc).__name__}")

    return (
        float(home_fraction), float(agent_density), int(seed),
        True, "", int(warmup_steps), time.perf_counter() - started,
        float((fighting / n_fans).mean()),
        float(fighting.mean()),
        float(fighting.max()),
        float(arrests.mean()),
        float(entropy.mean()),
        float((happy / n_fans).mean()),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_ofat"))
    parser.add_argument("--seed", type=int, default=DEFAULT_MODEL_SEED)
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (float(hf), float(ad), args.seed)
        for hf in HOME_FRACTIONS
        for ad in AGENT_DENSITIES
    ]

    metadata = {
        "sweep": {
            "home_fraction": [float(v) for v in HOME_FRACTIONS],
            "agent_density": [float(v) for v in AGENT_DENSITIES],
        },
        "fixed_parameters": {
            **FIXED,
            "home_fraction_aggressiveness_coupling": "native (aggressiveness_mean=None)",
            "hawk_dove_strategy": "logit_prior",
            "logit_beta": 5.0,
            "grid_N": 40,
            "perception_k": 0.693,
            "aggressiveness_concentration": 12.0,
        },
        "model_seed": args.seed,
        "riot_burn_in_steps": RIOT_BURN_IN_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "spatial_warmup": {
            "fine_entropy_cv_threshold": 0.025,
            "window": 10,
            "maximum_steps": MAX_SPATIAL_WARMUP_STEPS,
        },
        "model_runs": len(jobs),
        "workers": args.workers,
        "result_fields": list(RESULT_DTYPE.names),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Starting {len(jobs)} runs with {args.workers} workers, seed {args.seed}")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        raw = list(pool.map(run_one, jobs, chunksize=1))
    wall_seconds = time.perf_counter() - started

    results = np.asarray(raw, dtype=RESULT_DTYPE)
    results.sort(order=["home_fraction", "agent_density"])
    np.save(args.output_dir / "grid_results.npy", results, allow_pickle=False)

    invalid = int((~results["valid"]).sum())
    print(
        f"Finished in {wall_seconds:.1f}s; {invalid}/{len(results)} invalid; "
        f"data in {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
