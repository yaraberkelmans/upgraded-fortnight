"""Run the riot-model Sobol experiment with per-run snapshots and event files.

For four parameters and first-order Sobol sampling, the number of model runs is
``base_n * (4 + 2)``. The default base N is 32 (192 model runs).

Per completed run
-----------------
- ``data/runs/run_XXXX.npy``: 100 measurement snapshots.
- ``data/arrests/arrests_XXXX.npy``: one row per arrested fan during measurement.
- ``data/overview/overview_XXXX.npy``: one row with phase entropy diagnostics.

Each worker writes only files belonging to its own sample ID, so the layout is
safe with multiprocessing. After all workers finish, the main process combines
all overview files into ``data/run_overview.npy``.
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
from model.fan import FanGroup

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

DEFAULT_BASE_N = 32
DEFAULT_MODEL_SEED = 43
RIOT_BURN_IN_STEPS = 100
MEASUREMENT_STEPS = 100
MAX_SPATIAL_WARMUP_STEPS = 2_000

RESULT_DTYPE = np.dtype(
    [
        ("sample_id", np.int32),
        ("valid", np.bool_),
        ("failure_code", "U64"),
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
    ]
)

MEASUREMENT_DTYPE = np.dtype(
    [
        ("step", np.int16),
        ("fighting", np.int32),
        ("home", np.int32),
        ("away", np.int32),
        ("police", np.int32),
        ("happy", np.int32),
        ("unhappy", np.int32),
        ("moves", np.int32),
        ("arrests_step", np.int32),
        ("measurement_total_arrests", np.int32),
        ("spatial_entropy_local", np.float64),
        ("average_similarity", np.float64),
        ("happy_fraction", np.float64),
        ("fighting_fraction", np.float64),
        ("average_win_probability", np.float64),
        ("average_arrest_probability", np.float64),
    ]
)

# The array index is not used as time because this is a compact event table.
# ``step`` is the 0-based measurement snapshot in which the arrest happened.
ARREST_DTYPE = np.dtype(
    [
        ("step", np.int16),
        ("is_respawn", np.bool_),
        ("aggressiveness", np.float64),
        ("is_home", np.bool_),
    ]
)

OVERVIEW_DTYPE = np.dtype(
    [
        ("run_id", np.int32),
        ("valid", np.bool_),
        ("warmup_steps", np.int32),
        ("warmup_entropy", np.float64),
        ("start_measurement_entropy", np.float64),
        ("end_measurement_entropy", np.float64),
    ]
)


def _fine_entropy(model: RiotModel) -> float:
    return float(model._zone_entropy(model.segregation_params.zone_size_fine))


def _save_overview(
    overview_dir: Path,
    sample_id: int,
    valid: bool,
    warmup_steps: int,
    warmup_entropy: float,
    start_measurement_entropy: float,
    end_measurement_entropy: float,
) -> None:
    overview_dir.mkdir(parents=True, exist_ok=True)
    row = np.asarray(
        [
            (
                sample_id,
                valid,
                warmup_steps,
                warmup_entropy,
                start_measurement_entropy,
                end_measurement_entropy,
            )
        ],
        dtype=OVERVIEW_DTYPE,
    )
    np.save(overview_dir / f"overview_{sample_id:04d}.npy", row, allow_pickle=False)


def _failed(
    sample_id: int,
    values: tuple[float, ...],
    model_seed: int,
    warmup_steps: int,
    started: float,
    code: str,
) -> tuple:
    similarity_threshold, fight_threshold, hawk_dove_C, police_density = values
    nan = float("nan")
    return (
        sample_id,
        False,
        code,
        model_seed,
        float(similarity_threshold),
        float(fight_threshold),
        float(hawk_dove_C),
        float(police_density),
        warmup_steps,
        time.perf_counter() - started,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
    )


def run_one(job: tuple[int, tuple[float, ...], str, int]) -> tuple:
    sample_id, values, output_dir_str, model_seed = job
    output_dir = Path(output_dir_str)
    run_dir = output_dir / "runs"
    arrest_dir = output_dir / "arrests"
    overview_dir = output_dir / "overview"

    similarity_threshold, fight_threshold, hawk_dove_C, police_density = values
    started = time.perf_counter()
    nan = float("nan")

    segregation = SegregationParams(
        N=40,
        agent_density=0.6,
        home_fraction=0.50,
        similarity_threshold=float(similarity_threshold),
        movement_decay=1.0,
        seed=model_seed,
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
        _save_overview(overview_dir, sample_id, False, 0, nan, nan, nan)
        return _failed(
            sample_id, values, model_seed, 0, started, f"init:{type(exc).__name__}"
        )

    warmup_steps = 0
    try:
        while model.in_warmup and warmup_steps < MAX_SPATIAL_WARMUP_STEPS:
            model.step()
            warmup_steps += 1
    except Exception as exc:
        warmup_entropy = _fine_entropy(model)
        _save_overview(
            overview_dir,
            sample_id,
            False,
            warmup_steps,
            warmup_entropy,
            nan,
            nan,
        )
        return _failed(
            sample_id,
            values,
            model_seed,
            warmup_steps,
            started,
            f"warmup:{type(exc).__name__}",
        )

    warmup_entropy = _fine_entropy(model)
    if model.in_warmup:
        _save_overview(
            overview_dir,
            sample_id,
            False,
            warmup_steps,
            warmup_entropy,
            nan,
            nan,
        )
        return _failed(
            sample_id,
            values,
            model_seed,
            warmup_steps,
            started,
            "warmup_not_converged",
        )

    try:
        for _ in range(RIOT_BURN_IN_STEPS):
            model.step()

        start_measurement_entropy = _fine_entropy(model)
        measurement = np.empty(MEASUREMENT_STEPS, dtype=MEASUREMENT_DTYPE)
        arrest_records: list[tuple[int, bool, float, bool]] = []
        measurement_total_arrests = 0

        for t in range(MEASUREMENT_STEPS):
            model.step()

            # Police.arrest records only properties. The measurement step is
            # attached here, giving the compact tuple requested for analysis.
            for is_respawn, aggressiveness, is_home in model.arrested_fans_this_step:
                arrest_records.append(
                    (
                        t,
                        bool(is_respawn),
                        float(aggressiveness),
                        bool(is_home),
                    )
                )

            n_fans = max(len(model.fans), 1)
            home = model.count_group(FanGroup.HOME)
            away = model.count_group(FanGroup.AWAY)
            fighting = model.count_fighting_fans()
            happy = model.count_happy()
            arrests_step = int(model.arrests_this_step)
            measurement_total_arrests += arrests_step

            measurement[t] = (
                t,
                fighting,
                home,
                away,
                model.count_police(),
                happy,
                n_fans - happy,
                model.moves_this_step,
                arrests_step,
                measurement_total_arrests,
                _fine_entropy(model),
                model.average_similarity(),
                happy / n_fans,
                fighting / n_fans,
                model.average_perceived_win_probability(),
                model.average_perceived_arrest_probability(),
            )

        end_measurement_entropy = _fine_entropy(model)

        run_dir.mkdir(parents=True, exist_ok=True)
        arrest_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            run_dir / f"run_{sample_id:04d}.npy",
            measurement,
            allow_pickle=False,
        )
        np.save(
            arrest_dir / f"arrests_{sample_id:04d}.npy",
            np.asarray(arrest_records, dtype=ARREST_DTYPE),
            allow_pickle=False,
        )
        _save_overview(
            overview_dir,
            sample_id,
            True,
            warmup_steps,
            warmup_entropy,
            start_measurement_entropy,
            end_measurement_entropy,
        )
    except Exception as exc:
        _save_overview(
            overview_dir,
            sample_id,
            False,
            warmup_steps,
            warmup_entropy,
            locals().get("start_measurement_entropy", nan),
            nan,
        )
        return _failed(
            sample_id,
            values,
            model_seed,
            warmup_steps,
            started,
            f"riot:{type(exc).__name__}",
        )

    fighting = measurement["fighting"].astype(np.float64)
    arrests = measurement["arrests_step"].astype(np.float64)
    entropy = measurement["spatial_entropy_local"]
    similarity = measurement["average_similarity"]
    happy_fraction = measurement["happy_fraction"]
    win_probability = measurement["average_win_probability"]
    arrest_probability = measurement["average_arrest_probability"]
    n_fans = max(len(model.fans), 1)

    return (
        sample_id,
        True,
        "",
        model_seed,
        float(similarity_threshold),
        float(fight_threshold),
        float(hawk_dove_C),
        float(police_density),
        warmup_steps,
        time.perf_counter() - started,
        float(fighting.mean()),
        float(fighting.std()),
        float(fighting.max()),
        float((fighting / n_fans).mean()),
        float(arrests.sum()),
        float(arrests.mean()),
        float(entropy.mean()),
        float(similarity.mean()),
        float(happy_fraction.mean()),
        float(win_probability.mean()),
        float(arrest_probability.mean()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--base-n",
        type=int,
        default=DEFAULT_BASE_N,
        help="Sobol base sample size; total runs = base_n * 6.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_MODEL_SEED,
        help=(
            "Random seed used by every model run. The Sobol parameter sample "
            "remains fixed (sampling seed 2026)."
        ),
    )
    return parser.parse_args()


def _combine_overviews(output_dir: Path, expected_runs: int) -> None:
    paths = sorted((output_dir / "overview").glob("overview_*.npy"))
    if not paths:
        return

    combined = np.concatenate([np.load(path, allow_pickle=False) for path in paths])
    combined.sort(order="run_id")
    np.save(output_dir / "run_overview.npy", combined, allow_pickle=False)

    if len(combined) != expected_runs:
        print(
            f"WARNING: overview has {len(combined)} rows; " f"expected {expected_runs}."
        )


def main() -> None:
    args = parse_args()
    if args.base_n <= 0:
        raise ValueError("--base-n must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("runs", "arrests", "overview"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    samples = sobol_sample.sample(
        PROBLEM,
        args.base_n,
        calc_second_order=False,
        scramble=True,
        seed=2026,
    )
    expected_shape = (
        args.base_n * (PROBLEM["num_vars"] + 2),
        PROBLEM["num_vars"],
    )
    if samples.shape != expected_shape:
        raise RuntimeError(f"Unexpected Sobol sample shape: {samples.shape}")

    jobs = [
        (i, tuple(map(float, row)), str(args.output_dir), args.seed)
        for i, row in enumerate(samples)
    ]
    np.save(args.output_dir / "sobol_samples.npy", samples, allow_pickle=False)

    metadata = {
        "problem": PROBLEM,
        "base_n": args.base_n,
        "model_runs": len(jobs),
        "model_seed": args.seed,
        "sobol_sampling_seed": 2026,
        "fixed_parameters": {
            "home_fraction": 0.5,
            "hawk_dove_strategy": "logit_prior",
            "logit_beta": 5.0,
            "agent_density": 0.6,
            "grid_N": 40,
        },
        "spatial_warmup": {
            "fine_entropy_cv_threshold": 0.025,
            "window": 10,
            "maximum_steps": MAX_SPATIAL_WARMUP_STEPS,
        },
        "riot_burn_in_steps": RIOT_BURN_IN_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "workers": args.workers,
        "per_run_files": {
            "measurement": "runs/run_XXXX.npy",
            "arrests": "arrests/arrests_XXXX.npy",
            "overview": "overview/overview_XXXX.npy",
            "arrest_fields": list(ARREST_DTYPE.names),
            "overview_fields": list(OVERVIEW_DTYPE.names),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"Starting {len(jobs)} runs with {args.workers} workers "
        f"and model seed {args.seed}"
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        raw = list(pool.map(run_one, jobs, chunksize=1))
    wall_seconds = time.perf_counter() - started

    results = np.asarray(raw, dtype=RESULT_DTYPE)
    results.sort(order="sample_id")
    np.save(args.output_dir / "run_results.npy", results, allow_pickle=False)

    output_fields = [
        "mean_fighting",
        "std_fighting",
        "peak_fighting",
        "mean_fighting_fraction",
        "arrests_measurement",
        "mean_arrests_per_step",
        "mean_spatial_entropy_local",
        "mean_similarity",
        "mean_happy_fraction",
        "mean_win_probability",
        "mean_arrest_probability",
    ]
    np.savez_compressed(
        args.output_dir / "sobol_outputs.npz",
        **{field: results[field] for field in output_fields},
    )

    _combine_overviews(args.output_dir, len(jobs))

    timing = {
        "wall_seconds": wall_seconds,
        "mean_run_seconds": float(results["runtime_seconds"].mean()),
        "median_run_seconds": float(np.median(results["runtime_seconds"])),
        "max_run_seconds": float(results["runtime_seconds"].max()),
        "valid_runs": int(results["valid"].sum()),
        "invalid_runs": int((~results["valid"]).sum()),
    }
    (args.output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2),
        encoding="utf-8",
    )

    invalid = results[~results["valid"]]
    if invalid.size:
        np.save(args.output_dir / "invalid_runs.npy", invalid, allow_pickle=False)
        print(f"WARNING: {invalid.size} invalid runs. See invalid_runs.npy")

    print(f"Finished in {wall_seconds:.2f}s; data in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
