"""Run one complete riot-model experiment and save timing, arrays and plots.

Phases
------
1. Spatial warm-up until the model's fine-entropy CV criterion is met.
2. Fixed riot burn-in of 100 steps; no measurement rows are stored.
3. Fixed measurement period of 100 steps; one structured NumPy row per step.

Run from the directory containing this file:
    python run_example.py

Or as a package module from its parent directory:
    python -m riot_model_refactor_updated.run_example
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .riot_model import RiotModel, RiotParams, SegregationParams
    from .system_metrics import SystemStateRecorder, summarize_measurement
except ImportError:
    from riot_model import RiotModel, RiotParams, SegregationParams
    from system_metrics import SystemStateRecorder, summarize_measurement


# ---------------------------------------------------------------------------
# Experiment settings: edit these values for a different example run.
# ---------------------------------------------------------------------------
SPATIAL_PARAMS = SegregationParams(
    N=40,
    agent_density=0.80,
    home_fraction=0.50,
    similarity_threshold=0.30,
    movement_decay=1.0,
    seed=42,
    torus=True,
    count_empty_as_different=True,
    zone_size=10,
    zone_size_fine=4,
    warmup_cv_threshold=0.01,
    warmup_window=10,
    random_move_chance=0.005,
    collect_data=False,
)

RIOT_PARAMS = RiotParams(
    police_density=0.05,
    perception_k=0.693,
    fan_vision=2,
    fight_threshold=0.0,
    police_vision=5,
    logit_beta=5.0,
    hawk_dove_strategy="logit_prior",
    hawk_dove_C=4.0,
    aggressiveness_mean=None,
    aggressiveness_concentration=12.0,
    fighting_enabled=True,
)

RIOT_BURN_IN_STEPS = 100
MEASUREMENT_STEPS = 100
MAX_SPATIAL_WARMUP_STEPS = 2_000
OUTPUT_DIR = Path("run_example_output")


def save_line_plot(
    state: np.ndarray,
    fields: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save one figure containing one or more closely related time series."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = state["step"]
    for field in fields:
        ax.plot(x, state[field], label=field)
    ax.set_title(title)
    ax.set_xlabel("Measurement step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(fields) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_plots(state: np.ndarray, plot_dir: Path) -> list[Path]:
    """Create readable plots covering the recorded aggregate system metrics."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        (["home", "away", "n_fans"], "Fan population", "Number of fans", "01_population.png"),
        (["happy", "unhappy", "fighting"], "Fan states", "Number of fans", "02_fan_states.png"),
        (["moves", "arrests_step"], "Movement and arrests per step", "Count", "03_activity.png"),
        (["total_arrests"], "Cumulative arrests", "Total arrests", "04_total_arrests.png"),
        (
            ["spatial_entropy", "spatial_entropy_local", "avg_similarity"],
            "Spatial mixing and similarity",
            "Metric value",
            "05_spatial_metrics.png",
        ),
        (
            ["avg_aggressiveness", "avg_win_probability", "avg_arrest_probability"],
            "Average behavioural and perceived probabilities",
            "Average value",
            "06_behaviour_probabilities.png",
        ),
        (
            ["fighting_fraction", "home_fraction_current"],
            "Population fractions",
            "Fraction",
            "07_fractions.png",
        ),
        (
            ["avg_move_distance", "avg_move_distance_moved"],
            "Movement distance",
            "Average Chebyshev distance",
            "08_move_distance.png",
        ),
    ]

    paths: list[Path] = []
    for fields, title, ylabel, filename in plot_specs:
        path = plot_dir / filename
        save_line_plot(state, fields, title, ylabel, path)
        paths.append(path)
    return paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    init_start = time.perf_counter()
    model = RiotModel(
        segregation_params=SPATIAL_PARAMS,
        riot_params=RIOT_PARAMS,
    )
    initialization_seconds = time.perf_counter() - init_start

    # Phase 1: spatial warm-up. The model itself switches in_warmup to False
    # once its fine-entropy CV criterion (or zero-movement criterion) is met.
    warmup_start = time.perf_counter()
    spatial_warmup_steps = 0
    while model.in_warmup and spatial_warmup_steps < MAX_SPATIAL_WARMUP_STEPS:
        model.step()
        spatial_warmup_steps += 1
    spatial_warmup_seconds = time.perf_counter() - warmup_start

    if model.in_warmup:
        raise RuntimeError(
            "Spatial warm-up did not converge within "
            f"{MAX_SPATIAL_WARMUP_STEPS} steps."
        )

    # Phase 2: fixed riot burn-in; deliberately store no system-state rows.
    burn_in_start = time.perf_counter()
    for _ in range(RIOT_BURN_IN_STEPS):
        model.step()
    riot_burn_in_seconds = time.perf_counter() - burn_in_start

    # Phase 3: exactly 100 measured steps.
    recorder = SystemStateRecorder()
    measurement_start = time.perf_counter()
    for measurement_step in range(MEASUREMENT_STEPS):
        model.step()
        recorder.record(model, step=measurement_step, phase="measurement")
    measurement_seconds = time.perf_counter() - measurement_start

    state = recorder.to_array()
    if state.shape != (MEASUREMENT_STEPS,):
        raise RuntimeError(
            f"Expected {MEASUREMENT_STEPS} measurement rows, got {state.shape}."
        )

    saving_start = time.perf_counter()
    array_path = OUTPUT_DIR / "system_state_100_steps.npy"
    np.save(array_path, state, allow_pickle=False)

    summary = summarize_measurement(state)
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    parameters = {
        "segregation_params": asdict(SPATIAL_PARAMS),
        "riot_params": asdict(RIOT_PARAMS),
        "riot_burn_in_steps": RIOT_BURN_IN_STEPS,
        "measurement_steps": MEASUREMENT_STEPS,
        "max_spatial_warmup_steps": MAX_SPATIAL_WARMUP_STEPS,
    }
    # Enum values are not JSON serializable by default.
    parameters["riot_params"]["hawk_dove_strategy"] = (
        RIOT_PARAMS.hawk_dove_strategy.value
    )
    parameters_path = OUTPUT_DIR / "parameters.json"
    parameters_path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")

    plot_start = time.perf_counter()
    plot_paths = make_plots(state, OUTPUT_DIR / "plots")
    plotting_seconds = time.perf_counter() - plot_start
    saving_seconds = time.perf_counter() - saving_start

    total_seconds = time.perf_counter() - total_start
    timings = {
        "initialization_seconds": initialization_seconds,
        "spatial_warmup_seconds": spatial_warmup_seconds,
        "spatial_warmup_steps": spatial_warmup_steps,
        "final_spatial_entropy_cv": float(model.entropy_cv_fine()),
        "riot_burn_in_seconds": riot_burn_in_seconds,
        "riot_burn_in_steps": RIOT_BURN_IN_STEPS,
        "measurement_seconds": measurement_seconds,
        "measurement_steps": MEASUREMENT_STEPS,
        "mean_measurement_step_seconds": measurement_seconds / MEASUREMENT_STEPS,
        "saving_and_plotting_seconds": saving_seconds,
        "plotting_seconds": plotting_seconds,
        "total_seconds": total_seconds,
    }
    timings_path = OUTPUT_DIR / "timings.json"
    timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")

    manifest = {
        "array": str(array_path),
        "summary": str(summary_path),
        "parameters": str(parameters_path),
        "timings": str(timings_path),
        "plots": [str(path) for path in plot_paths],
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Run completed successfully")
    print(f"Spatial warm-up steps: {spatial_warmup_steps}")
    print(f"Riot burn-in steps: {RIOT_BURN_IN_STEPS}")
    print(f"Measurement rows: {state.shape[0]}")
    print(f"Total execution time: {total_seconds:.3f} s")
    print(f"NumPy array: {array_path.resolve()}")
    print(f"Plots: {(OUTPUT_DIR / 'plots').resolve()}")


if __name__ == "__main__":
    main()
