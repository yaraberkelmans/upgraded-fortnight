"""
DATE: 28-6-2026
NAMES: Ruben, Mark, Yara, Max

Description: 
Run a small fixed-cohort arrest experiment. 
The experiment runs three setups, each with a specified set of parameters, 
and measures the time to arrest for a cohort of fans. 
The results are saved in a structured format for further analysis.


Disclaimer: 
AI may be used in with creating the code. 
We checked the code on functionality, logic and correctness. 
We are responsible for the code and its content.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from model.fan import FanGroup
from model.riot_model import RiotModel, RiotParams, SegregationParams

RIOT_BURN_IN_STEPS = 100
MEASUREMENT_STEPS = 100
MAX_SPATIAL_WARMUP_STEPS = 2_000
AGENT_DENSITY = 0.6
GRID_N = 40

COHORT_DTYPE = np.dtype(
    [
        ("cohort_id", np.int32),
        ("arrest_step", np.int16),
        ("is_respawn_at_start", np.bool_),
        ("aggressiveness_at_start", np.float64),
        ("is_home", np.bool_),
    ]
)

RUN_SUMMARY_DTYPE = np.dtype(
    [
        ("setup_id", np.int16),
        ("setup_name", "U64"),
        ("repeat", np.int16),
        ("seed", np.int64),
        ("valid", np.bool_),
        ("failure_code", "U64"),
        ("warmup_steps", np.int32),
        ("cohort_size", np.int32),
        ("arrested_count", np.int32),
        ("not_arrested_count", np.int32),
        ("arrest_fraction", np.float64),
        ("mean_time_to_arrest", np.float64),
        ("median_time_to_arrest", np.float64),
        ("runtime_seconds", np.float64),
    ]
)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "setup"


def load_setups(path: Path) -> list[dict]:
    setups = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(setups, list) or len(setups) != 3:
        raise ValueError("setups.json must contain exactly three setup objects")

    required = {
        "name",
        "similarity_threshold",
        "fight_threshold",
        "hawk_dove_C",
        "police_density",
        "logit_beta",
    }
    names: set[str] = set()
    for setup in setups:
        missing = required.difference(setup)
        if missing:
            raise ValueError(f"Setup is missing fields: {sorted(missing)}")
        setup["name"] = safe_name(str(setup["name"]))
        if setup["name"] in names:
            raise ValueError(f"Duplicate setup name: {setup['name']}")
        names.add(setup["name"])
    return setups


def failed_summary(
    setup_id: int,
    setup_name: str,
    repeat: int,
    seed: int,
    warmup_steps: int,
    started: float,
    code: str,
) -> tuple:
    nan = float("nan")
    return (
        setup_id,
        setup_name,
        repeat,
        seed,
        False,
        code,
        warmup_steps,
        0,
        0,
        0,
        nan,
        nan,
        nan,
        time.perf_counter() - started,
    )


def run_one(job: tuple[int, dict, int, int, str]) -> tuple:
    setup_id, setup, repeat, seed, output_dir_str = job
    output_dir = Path(output_dir_str)
    setup_name = setup["name"]
    setup_dir = output_dir / "cohort" / setup_name
    started = time.perf_counter()

    segregation = SegregationParams(
        N=GRID_N,
        agent_density=AGENT_DENSITY,
        home_fraction=0.50,
        similarity_threshold=float(setup["similarity_threshold"]),
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
        police_density=float(setup["police_density"]),
        perception_k=0.693,
        fan_vision=2,
        fight_threshold=float(setup["fight_threshold"]),
        police_vision=5,
        logit_beta=float(setup["logit_beta"]),
        hawk_dove_strategy="logit_qre",
        hawk_dove_C=float(setup["hawk_dove_C"]),
        aggressiveness_mean=None,
        aggressiveness_concentration=12.0,
        fighting_enabled=True,
    )

    try:
        model = RiotModel(segregation_params=segregation, riot_params=riot)
    except Exception as exc:
        return failed_summary(
            setup_id,
            setup_name,
            repeat,
            seed,
            0,
            started,
            f"init:{type(exc).__name__}",
        )

    warmup_steps = 0
    try:
        while model.in_warmup and warmup_steps < MAX_SPATIAL_WARMUP_STEPS:
            model.step()
            warmup_steps += 1
    except Exception as exc:
        return failed_summary(
            setup_id,
            setup_name,
            repeat,
            seed,
            warmup_steps,
            started,
            f"warmup:{type(exc).__name__}",
        )

    if model.in_warmup:
        return failed_summary(
            setup_id,
            setup_name,
            repeat,
            seed,
            warmup_steps,
            started,
            "warmup_not_converged",
        )

    try:
        for _ in range(RIOT_BURN_IN_STEPS):
            model.step()

        cohort = np.empty(len(model.fans), dtype=COHORT_DTYPE)
        for cohort_id, fan in enumerate(model.fans):
            fan.measurement_cohort_id = cohort_id
            cohort[cohort_id] = (
                cohort_id,
                -1,
                bool(fan.is_respawn),
                float(fan.aggressiveness),
                fan.group == FanGroup.HOME,
            )

        fighting_over_time = np.zeros(MEASUREMENT_STEPS, dtype=np.int32)
        for step in range(MEASUREMENT_STEPS):
            model.step()
            fighting_over_time[step] = int(model.count_fighting_fans())
            for cohort_id in model.arrested_fans_this_step:
                if cohort[cohort_id]["arrest_step"] == -1:
                    cohort[cohort_id]["arrest_step"] = step

        setup_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            setup_dir / f"cohort_repeat_{repeat:02d}_seed_{seed}.npy",
            cohort,
            allow_pickle=False,
        )
        fighting_dir = output_dir / "fighting" / setup_name
        fighting_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            fighting_dir / f"fighting_repeat_{repeat:02d}_seed_{seed}.npy",
            fighting_over_time,
            allow_pickle=False,
        )
    except Exception as exc:
        return failed_summary(
            setup_id,
            setup_name,
            repeat,
            seed,
            warmup_steps,
            started,
            f"riot:{type(exc).__name__}",
        )

    arrested_steps = cohort["arrest_step"]
    arrested = arrested_steps >= 0
    arrested_count = int(arrested.sum())
    cohort_size = int(len(cohort))
    not_arrested_count = cohort_size - arrested_count
    if arrested_count:
        mean_tta = float(arrested_steps[arrested].mean())
        median_tta = float(np.median(arrested_steps[arrested]))
    else:
        mean_tta = float("nan")
        median_tta = float("nan")

    return (
        setup_id,
        setup_name,
        repeat,
        seed,
        True,
        "",
        warmup_steps,
        cohort_size,
        arrested_count,
        not_arrested_count,
        arrested_count / cohort_size if cohort_size else float("nan"),
        mean_tta,
        median_tta,
        time.perf_counter() - started,
    )


def save_csv(path: Path, results: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(results.dtype.names)
        for row in results:
            writer.writerow([row[name].item() for name in results.dtype.names])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setups",
        type=Path,
        default=Path(__file__).with_name("setups.json"),
        help="JSON file containing exactly three setups.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("three_setup_data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    setups = load_setups(args.setups)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The same repetition seed is used for all three setups. This creates a
    # paired design: repeat 0 of each setup starts from the same model seed.
    jobs = []
    for setup_id, setup in enumerate(setups):
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            jobs.append((setup_id, setup, repeat, seed, str(args.output_dir)))

    metadata = {
        "experiment": "fixed_cohort_time_to_arrest",
        "number_of_setups": len(setups),
        "repeats_per_setup": args.repeats,
        "total_runs": len(jobs),
        "base_seed": args.seed,
        "paired_seeds": True,
        "agent_density": AGENT_DENSITY,
        "grid_N": GRID_N,
        "riot_burn_in_steps": RIOT_BURN_IN_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "setups": setups,
        "fighting_series": ("Number of fighting fans after each measurement step, saved per run."),
        "cohort_rule": (
            "Only fans present at measurement start are included. "
            "Respawns created during measurement are excluded."
        ),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Starting {len(jobs)} runs: {len(setups)} setups x "
        f"{args.repeats} repeats, using {args.workers} workers"
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        raw = list(pool.map(run_one, jobs, chunksize=1))

    results = np.asarray(raw, dtype=RUN_SUMMARY_DTYPE)
    results.sort(order=["setup_id", "repeat"])
    np.save(args.output_dir / "run_summary.npy", results, allow_pickle=False)
    save_csv(args.output_dir / "run_summary.csv", results)

    elapsed = time.perf_counter() - started
    invalid = results[~results["valid"]]
    print(f"Finished in {elapsed:.2f}s")
    print(f"Valid runs: {int(results['valid'].sum())}/{len(results)}")
    if invalid.size:
        np.save(args.output_dir / "invalid_runs.npy", invalid, allow_pickle=False)
        print("Some runs failed; see invalid_runs.npy")
    print(f"Output: {args.output_dir.resolve()}")
    print("Next: python plot_time_to_arrest.py --data-dir", args.output_dir)


if __name__ == "__main__":
    main()
