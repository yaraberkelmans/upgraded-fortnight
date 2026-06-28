"""
DATE: 28-6-2026
NAMES: Ruben, Mark, Yara, Max

Description: Create histogram and point plots for the three similarity-threshold setups. The plots visualize the time to arrest for agents present at measurement start, mean time to arrest, arrest fraction, and non-arrested count per run. Additionally, the mean number of fighting fans over time is plotted with ±1 standard deviation.
Disclaimer: AI may be used in with creating the code. We checked the code on functionality, logic and correctness. We are responsible for the code and its content.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MEASUREMENT_STEPS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("three_setup_data"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins across the measurement period.",
    )
    return parser.parse_args()


def load_setup(
    paths: list[Path],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Return arrest times, mean TTA, arrest fraction, and non-arrest count per run."""
    times_per_run: list[np.ndarray] = []
    mean_tta = []
    arrest_fraction = []
    not_arrested_count = []

    for path in paths:
        cohort = np.load(path, allow_pickle=False)
        steps = cohort["arrest_step"].astype(int)
        arrested_steps = steps[steps >= 0]
        times_per_run.append(arrested_steps)
        mean_tta.append(float(arrested_steps.mean()) if arrested_steps.size else np.nan)
        arrest_fraction.append(float(arrested_steps.size / len(cohort)) if len(cohort) else np.nan)
        not_arrested_count.append(int(len(cohort) - arrested_steps.size))

    return (
        times_per_run,
        np.asarray(mean_tta),
        np.asarray(arrest_fraction),
        np.asarray(not_arrested_count, dtype=int),
    )


def load_fighting_series(paths: list[Path]) -> np.ndarray:
    """Load per-run fighting counts into shape (runs, measurement_steps)."""
    series = []
    for path in paths:
        values = np.load(path, allow_pickle=False).astype(float)
        if values.ndim != 1 or len(values) != MEASUREMENT_STEPS:
            raise ValueError(f"Unexpected fighting series shape in {path}: {values.shape}")
        series.append(values)
    if not series:
        return np.empty((0, MEASUREMENT_STEPS), dtype=float)
    return np.vstack(series)


def plot_mean_fighting_over_time(
    setups: list[dict],
    fighting_by_setup: list[np.ndarray],
    output_path: Path,
) -> None:
    """Plot mean number of fighting fans over measurement time with ±1 SD."""
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(MEASUREMENT_STEPS)

    for setup, values in zip(setups, fighting_by_setup):
        if values.size == 0:
            continue
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        label = f"threshold={setup['similarity_threshold']:.2f}"
        line = ax.plot(steps, mean, linewidth=2, label=label)[0]
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            alpha=0.18,
            color=line.get_color(),
        )

    ax.set_xlabel("Measurement step")
    ax.set_ylabel("Mean number of fighting fans")
    ax.set_title("Mean Fighting Over Time by Similarity Threshold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def threshold_label(setup: dict) -> str:
    return f"{setup['name'].replace('_', ' ')}\nthreshold={setup['similarity_threshold']:.2f}"


def plot_histograms(
    setup_data: list[tuple[dict, list[np.ndarray], np.ndarray, np.ndarray]],
    output_path: Path,
    bins: int,
) -> None:
    """Plot arrest times as a percentage of all agents present at measurement start.

    Agents that are never arrested are not assigned a time-to-arrest bin. Their
    share is reported explicitly in each panel. Consequently, the histogram bars
    sum to the percentage of starting agents that were arrested.
    """
    edges = np.linspace(0, MEASUREMENT_STEPS, bins + 1)
    fig, axes = plt.subplots(1, len(setup_data), figsize=(15, 5.5), sharex=True, sharey=True)
    if len(setup_data) == 1:
        axes = [axes]

    for ax, (setup, times_per_run, arrest_fraction, not_arrested_count) in zip(axes, setup_data):
        pooled = (
            np.concatenate([x for x in times_per_run if x.size])
            if any(x.size for x in times_per_run)
            else np.array([], dtype=int)
        )
        total_starting_agents = int(sum(len(x) for x in times_per_run) + np.sum(not_arrested_count))
        arrested_count = int(pooled.size)
        included_pct = (
            (100.0 * arrested_count / total_starting_agents) if total_starting_agents else 0.0
        )
        not_arrested_pct = 100.0 - included_pct if total_starting_agents else 0.0

        weights = (
            np.full(arrested_count, 100.0 / total_starting_agents)
            if total_starting_agents and arrested_count
            else None
        )
        ax.hist(
            pooled,
            bins=edges,
            weights=weights,
            alpha=0.75,
            edgecolor="black",
        )
        if pooled.size:
            ax.axvline(
                float(np.mean(pooled)),
                linestyle="--",
                linewidth=1.5,
                label=f"Mean arrest step = {np.mean(pooled):.1f}",
            )
            ax.legend(fontsize=8)

        ax.text(
            0.98,
            0.97,
            f"Included (arrested): {included_pct:.1f}%\nNot arrested: {not_arrested_pct:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
        )
        ax.set_title(threshold_label(setup))
        ax.set_xlabel("Time to arrest (measurement step)")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Percentage of agents present at measurement start")
    fig.suptitle(
        "Time to Arrest by Similarity Threshold for Agents Present at "
        "Measurement Start (after warmup)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def jitter_positions(center: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([center])
    return center + np.linspace(-0.12, 0.12, count)


def plot_points(
    setups: list[dict],
    values_by_setup: list[np.ndarray],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for index, (setup, values) in enumerate(zip(setups, values_by_setup)):
        finite = values[np.isfinite(values)]
        x = jitter_positions(float(index), len(finite))
        ax.scatter(x, finite, s=45, alpha=0.75, label="Individual run" if index == 0 else None)
        if finite.size:
            mean = float(np.mean(finite))
            std = float(np.std(finite))
            ax.errorbar(
                index,
                mean,
                yerr=std,
                fmt="D",
                markersize=7,
                capsize=5,
                linewidth=1.8,
                label="Mean ± SD" if index == 0 else None,
            )

    ax.set_xticks(range(len(setups)))
    ax.set_xticklabels([f"{s['similarity_threshold']:.2f}" for s in setups])
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    output_dir = args.output_dir or (args.data_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.data_dir / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    setups: list[dict] = []
    histogram_data: list[tuple[dict, list[np.ndarray], np.ndarray, np.ndarray]] = []
    mean_tta_by_setup: list[np.ndarray] = []
    arrest_fraction_by_setup: list[np.ndarray] = []
    not_arrested_count_by_setup: list[np.ndarray] = []
    fighting_by_setup: list[np.ndarray] = []
    export: dict[str, np.ndarray] = {}

    for setup in metadata["setups"]:
        name = setup["name"]
        paths = sorted((args.data_dir / "cohort" / name).glob("cohort_*.npy"))
        if not paths:
            print(f"Skipping {name}: no cohort files")
            continue

        times_per_run, mean_tta, arrest_fraction, not_arrested_count = load_setup(paths)
        fighting_paths = sorted((args.data_dir / "fighting" / name).glob("fighting_*.npy"))
        fighting_series = load_fighting_series(fighting_paths)
        setups.append(setup)
        histogram_data.append((setup, times_per_run, arrest_fraction, not_arrested_count))
        mean_tta_by_setup.append(mean_tta)
        arrest_fraction_by_setup.append(arrest_fraction)
        not_arrested_count_by_setup.append(not_arrested_count)
        fighting_by_setup.append(fighting_series)

        pooled = (
            np.concatenate([x for x in times_per_run if x.size])
            if any(x.size for x in times_per_run)
            else np.array([], dtype=int)
        )
        export[f"{name}__pooled_arrest_steps"] = pooled
        export[f"{name}__mean_tta_per_run"] = mean_tta
        export[f"{name}__arrest_fraction_per_run"] = arrest_fraction
        export[f"{name}__not_arrested_count_per_run"] = not_arrested_count
        export[f"{name}__fighting_over_time_per_run"] = fighting_series
        if fighting_series.size:
            export[f"{name}__mean_fighting_over_time"] = np.mean(fighting_series, axis=0)
            export[f"{name}__std_fighting_over_time"] = np.std(fighting_series, axis=0)

    if not setups:
        raise SystemExit("No cohort files found")

    plot_histograms(
        histogram_data,
        output_dir / "time_to_arrest_histograms.png",
        args.bins,
    )
    plot_points(
        setups,
        mean_tta_by_setup,
        "Mean time to arrest within a run",
        "Mean time to arrest for each run",
        output_dir / "mean_time_to_arrest_points.png",
    )
    plot_points(
        setups,
        arrest_fraction_by_setup,
        "Fraction of starting agents arrested",
        "Arrest fraction for each run",
        output_dir / "arrest_fraction_points.png",
    )
    plot_points(
        setups,
        not_arrested_count_by_setup,
        "Number of starting agents not arrested",
        "Non-arrested starting agents for each run",
        output_dir / "not_arrested_count_points.png",
    )
    plot_mean_fighting_over_time(
        setups,
        fighting_by_setup,
        output_dir / "mean_fighting_over_time.png",
    )

    np.savez_compressed(args.data_dir / "time_to_arrest_plot_data.npz", **export)

    print("\nSummary by similarity threshold:")
    for setup, mean_tta, arrest_fraction, not_arrested_count in zip(
        setups, mean_tta_by_setup, arrest_fraction_by_setup, not_arrested_count_by_setup
    ):
        finite_tta = mean_tta[np.isfinite(mean_tta)]
        mean_tta_value = float(np.mean(finite_tta)) if finite_tta.size else float("nan")
        print(
            f"  threshold={setup['similarity_threshold']:.2f}: "
            f"runs={len(mean_tta)}, "
            f"mean TTA={mean_tta_value:.2f}, "
            f"mean arrest fraction={np.nanmean(arrest_fraction):.3%}, "
            f"mean not arrested={np.mean(not_arrested_count):.1f}"
        )
    print(f"\nPlots saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
