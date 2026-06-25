"""Run a 4-parameter first-order Sobol pilot with exactly 6 * 64 = 384 runs.

Phases per model run
--------------------
1. Spatial warm-up until the model's fine-entropy CV criterion exits.
2. Fixed riot burn-in of 100 steps; no output retained.
3. Fixed measurement period of 100 steps; scalar summaries retained.

The model package in ./model is imported without modification.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from SALib.sample import sobol as sobol_sample

from model.riot_model import RiotModel, RiotParams, SegregationParams

PROBLEM = {
    "num_vars": 4,
    "names": [
        "similarity_threshold",
        "fight_threshold",
        "hawk_dove_C",
        "police_density",
    ],
    "bounds": [
        [0.05, 0.90],
        [-0.35, 0.45],
        [0.10, 19.00],
        [0.01, 0.15],
    ],
}

BASE_N = 64
FIXED_SEED = 42
RIOT_BURN_IN_STEPS = 100
MEASUREMENT_STEPS = 100
MAX_SPATIAL_WARMUP_STEPS = 2_000

RESULT_DTYPE = np.dtype([
    ("sample_id", np.int32),
    ("valid", np.bool_),
    ("failure_code", "U32"),
    ("seed", np.int64),
    ("similarity_threshold", np.float64),
    ("fight_threshold", np.float64),
    ("hawk_dove_C", np.float64),
    ("police_density", np.float64),
    ("warmup_steps", np.int32),
    ("runtime_seconds", np.float64),
    ("mean_fighting", np.float64),
    ("std_fighting", np.float64),
    ("peak_fighting", np.float64),
    ("mean_fighting_fraction", np.float64),
    ("arrests_measurement", np.float64),
    ("mean_arrests_per_step", np.float64),
    ("mean_spatial_entropy_local", np.float64),
    ("mean_similarity", np.float64),
    ("mean_happy_fraction", np.float64),
    ("mean_win_probability", np.float64),
    ("mean_arrest_probability", np.float64),
])


def run_one(job: tuple[int, tuple[float, ...]]) -> tuple:
    sample_id, values = job
    similarity_threshold, fight_threshold, hawk_dove_C, police_density = values
    started = time.perf_counter()

    segregation = SegregationParams(
        N=40,
        agent_density=0.80,
        home_fraction=0.50,
        similarity_threshold=float(similarity_threshold),
        movement_decay=1.0,
        seed=FIXED_SEED,
        torus=True,
        count_empty_as_different=True,
        zone_size=10,
        zone_size_fine=4,
        warmup_cv_threshold=0.01,
        warmup_window=10,
        random_move_chance=0.005,
        collect_data=False,
    )
    riot = RiotParams(
        police_density=float(police_density),
        perception_k=0.693,
        fan_vision=2,
        fight_threshold=float(fight_threshold),
        police_vision=5,
        logit_beta=5.0,
        hawk_dove_strategy="logit_prior",
        hawk_dove_C=float(hawk_dove_C),
        aggressiveness_mean=None,
        aggressiveness_concentration=12.0,
        fighting_enabled=True,
    )

    try:
        model = RiotModel(segregation_params=segregation, riot_params=riot)
    except Exception as exc:
        return _failed(sample_id, values, 0, started, f"init:{type(exc).__name__}")

    warmup_steps = 0
    try:
        while model.in_warmup and warmup_steps < MAX_SPATIAL_WARMUP_STEPS:
            model.step()
            warmup_steps += 1
    except Exception as exc:
        return _failed(sample_id, values, warmup_steps, started, f"warmup:{type(exc).__name__}")

    if model.in_warmup:
        return _failed(sample_id, values, warmup_steps, started, "warmup_not_converged")

    try:
        for _ in range(RIOT_BURN_IN_STEPS):
            model.step()

        fighting = np.empty(MEASUREMENT_STEPS, dtype=np.float64)
        arrests = np.empty(MEASUREMENT_STEPS, dtype=np.float64)
        entropy = np.empty(MEASUREMENT_STEPS, dtype=np.float64)
        similarity = np.empty(MEASUREMENT_STEPS, dtype=np.float64)
        happy_fraction = np.empty(MEASUREMENT_STEPS, dtype=np.float64)
        win_probability = np.empty(MEASUREMENT_STEPS, dtype=np.float64)
        arrest_probability = np.empty(MEASUREMENT_STEPS, dtype=np.float64)

        for t in range(MEASUREMENT_STEPS):
            model.step()
            n_fans = max(len(model.fans), 1)
            fighting[t] = model.count_fighting_fans()
            arrests[t] = model.arrests_this_step
            entropy[t] = model._zone_entropy(model.segregation_params.zone_size_fine)
            similarity[t] = model.average_similarity()
            happy_fraction[t] = model.count_happy() / n_fans
            win_probability[t] = model.average_perceived_win_probability()
            arrest_probability[t] = model.average_perceived_arrest_probability()
    except Exception as exc:
        return _failed(sample_id, values, warmup_steps, started, f"riot:{type(exc).__name__}")

    n_fans = max(len(model.fans), 1)
    return (
        sample_id, True, "", FIXED_SEED,
        float(similarity_threshold), float(fight_threshold),
        float(hawk_dove_C), float(police_density),
        warmup_steps, time.perf_counter() - started,
        float(fighting.mean()), float(fighting.std()), float(fighting.max()),
        float((fighting / n_fans).mean()),
        float(arrests.sum()), float(arrests.mean()),
        float(entropy.mean()), float(similarity.mean()),
        float(happy_fraction.mean()), float(win_probability.mean()),
        float(arrest_probability.mean()),
    )


def _failed(sample_id, values, warmup_steps, started, code):
    similarity_threshold, fight_threshold, hawk_dove_C, police_density = values
    nan = float("nan")
    return (
        sample_id, False, code, FIXED_SEED,
        float(similarity_threshold), float(fight_threshold),
        float(hawk_dove_C), float(police_density),
        warmup_steps, time.perf_counter() - started,
        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = sobol_sample.sample(
        PROBLEM,
        BASE_N,
        calc_second_order=False,
        scramble=True,
        seed=2026,
    )
    if samples.shape != (BASE_N * (PROBLEM["num_vars"] + 2), PROBLEM["num_vars"]):
        raise RuntimeError(f"Unexpected Sobol sample shape: {samples.shape}")

    jobs = [(i, tuple(map(float, row))) for i, row in enumerate(samples)]
    np.save(args.output_dir / "sobol_samples.npy", samples, allow_pickle=False)

    metadata = {
        "problem": PROBLEM,
        "base_n": BASE_N,
        "model_runs": len(jobs),
        "fixed_seed": FIXED_SEED,
        "fixed_parameters": {
            "home_fraction": 0.5,
            "hawk_dove_strategy": "logit_prior",
            "logit_beta": 5.0,
            "agent_density": 0.8,
            "grid_N": 40,
        },
        "spatial_warmup": {
            "fine_entropy_cv_threshold": 0.01,
            "window": 10,
            "maximum_steps": MAX_SPATIAL_WARMUP_STEPS,
        },
        "riot_burn_in_steps": RIOT_BURN_IN_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "workers": args.workers,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Starting {len(jobs)} runs with {args.workers} workers")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        raw = list(pool.map(run_one, jobs, chunksize=1))
    wall_seconds = time.perf_counter() - started

    results = np.asarray(raw, dtype=RESULT_DTYPE)
    results.sort(order="sample_id")
    np.save(args.output_dir / "run_results.npy", results, allow_pickle=False)

    output_fields = [
        "mean_fighting", "std_fighting", "peak_fighting",
        "mean_fighting_fraction", "arrests_measurement",
        "mean_arrests_per_step", "mean_spatial_entropy_local",
        "mean_similarity", "mean_happy_fraction",
        "mean_win_probability", "mean_arrest_probability",
    ]
    np.savez_compressed(
        args.output_dir / "sobol_outputs.npz",
        **{field: results[field] for field in output_fields},
    )

    timing = {
        "wall_seconds": wall_seconds,
        "mean_run_seconds": float(results["runtime_seconds"].mean()),
        "median_run_seconds": float(np.median(results["runtime_seconds"])),
        "max_run_seconds": float(results["runtime_seconds"].max()),
        "valid_runs": int(results["valid"].sum()),
        "invalid_runs": int((~results["valid"]).sum()),
    }
    (args.output_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    invalid = results[~results["valid"]]
    if invalid.size:
        np.save(args.output_dir / "invalid_runs.npy", invalid, allow_pickle=False)
        print(f"WARNING: {invalid.size} invalid runs. See invalid_runs.npy")
    print(f"Finished in {wall_seconds:.2f}s; data in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
