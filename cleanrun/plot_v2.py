"""Plot V2: inspect fighting outcomes, parameter interactions, and instability.

Works with the per-run Sobol folder structure used for both N=32 and N=64:

data/
├── runs/run_0000.npy
├── runs/run_0001.npy
├── ...
├── sobol_samples.npy
└── run_results.npy          # optional but recommended

The script reads every available per-run file, combines it with its sampled
parameter values, calculates stability diagnostics, and creates:

1. One 3D interaction plot for every pair of parameters.
   x/y = parameter pair, z = mean fighting fans, colour = instability score.
2. A parameter-versus-instability panel.
3. Mean versus standard deviation of fighting.
4. Peak versus mean fighting.
5. Fighting time series for the most unstable runs.
6. A CSV and NPY table with all calculated diagnostics.

Usage
-----
python plot_v2.py
python plot_v2.py --data-dir data --plots-dir plots_v2
python plot_v2.py --top-unstable 12 --instability-quantile 0.90
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PARAMETER_NAMES = [
    "similarity_threshold",
    "fight_threshold",
    "hawk_dove_C",
    "police_density",
]

PARAMETER_LABELS = {
    "similarity_threshold": "Similarity threshold",
    "fight_threshold": "Fight threshold",
    "hawk_dove_C": "Hawk–dove C",
    "police_density": "Police density",
}

SUMMARY_DTYPE = np.dtype(
    [
        ("sample_id", np.int32),
        ("similarity_threshold", np.float64),
        ("fight_threshold", np.float64),
        ("hawk_dove_C", np.float64),
        ("police_density", np.float64),
        ("n_steps", np.int32),
        ("mean_fighting", np.float64),
        ("std_fighting", np.float64),
        ("peak_fighting", np.float64),
        ("min_fighting", np.float64),
        ("fighting_range", np.float64),
        ("cv_fighting", np.float64),
        ("max_step_jump", np.float64),
        ("mean_abs_step_jump", np.float64),
        ("early_mean_fighting", np.float64),
        ("late_mean_fighting", np.float64),
        ("absolute_drift", np.float64),
        ("relative_drift", np.float64),
        ("fraction_zero_fighting", np.float64),
        ("mean_fighting_fraction", np.float64),
        ("mean_happy_fraction", np.float64),
        ("mean_entropy", np.float64),
        ("mean_arrests", np.float64),
        ("instability_score", np.float64),
        ("unstable_flag", np.bool_),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create interaction and instability plots for Sobol runs."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots_v2"))
    parser.add_argument(
        "--top-unstable",
        type=int,
        default=10,
        help="Number of most unstable time series to plot.",
    )
    parser.add_argument(
        "--instability-quantile",
        type=float,
        default=0.90,
        help="Runs at or above this instability quantile are flagged.",
    )
    return parser.parse_args()


def run_id_from_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Cannot read run ID from {path.name}") from exc


def robust_scale(values: np.ndarray) -> np.ndarray:
    """Median/IQR scaling, robust to a few extreme runs."""
    values = np.asarray(values, dtype=float)
    median = float(np.nanmedian(values))
    q25, q75 = np.nanpercentile(values, [25, 75])
    iqr = float(q75 - q25)

    if not np.isfinite(iqr) or iqr <= 1e-12:
        std = float(np.nanstd(values))
        iqr = std if std > 1e-12 else 1.0

    return np.maximum((values - median) / iqr, 0.0)


def calculate_raw_metrics(
    run: np.ndarray,
    sample_id: int,
    params: np.ndarray,
) -> dict[str, Any]:
    fields = set(run.dtype.names or ())
    required = {"fighting"}
    missing = required - fields
    if missing:
        raise ValueError(
            f"run_{sample_id:04d}.npy misses required fields: {sorted(missing)}"
        )

    fighting = np.asarray(run["fighting"], dtype=float)
    if fighting.size == 0:
        raise ValueError(f"run_{sample_id:04d}.npy contains no measurement rows")
    if not np.all(np.isfinite(fighting)):
        raise ValueError(f"run_{sample_id:04d}.npy contains non-finite fighting values")

    differences = np.diff(fighting)
    split = max(1, fighting.size // 2)
    early = fighting[:split]
    late = fighting[split:]
    if late.size == 0:
        late = fighting[-1:]

    mean_fighting = float(np.mean(fighting))
    std_fighting = float(np.std(fighting))
    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    absolute_drift = abs(late_mean - early_mean)

    denominator = max(abs(mean_fighting), 1.0)
    cv_fighting = std_fighting / denominator
    relative_drift = absolute_drift / denominator

    n_fans = None
    if "home" in fields and "away" in fields:
        fans = np.asarray(run["home"], dtype=float) + np.asarray(
            run["away"], dtype=float
        )
        if np.any(fans > 0):
            n_fans = np.maximum(fans, 1.0)

    if "fighting_fraction" in fields:
        mean_fighting_fraction = float(np.nanmean(run["fighting_fraction"]))
    elif n_fans is not None:
        mean_fighting_fraction = float(np.mean(fighting / n_fans))
    else:
        mean_fighting_fraction = float("nan")

    def optional_mean(field: str) -> float:
        if field not in fields:
            return float("nan")
        values = np.asarray(run[field], dtype=float)
        return float(np.nanmean(values))

    return {
        "sample_id": sample_id,
        "similarity_threshold": float(params[0]),
        "fight_threshold": float(params[1]),
        "hawk_dove_C": float(params[2]),
        "police_density": float(params[3]),
        "n_steps": int(fighting.size),
        "mean_fighting": mean_fighting,
        "std_fighting": std_fighting,
        "peak_fighting": float(np.max(fighting)),
        "min_fighting": float(np.min(fighting)),
        "fighting_range": float(np.ptp(fighting)),
        "cv_fighting": cv_fighting,
        "max_step_jump": float(np.max(np.abs(differences))) if differences.size else 0.0,
        "mean_abs_step_jump": (
            float(np.mean(np.abs(differences))) if differences.size else 0.0
        ),
        "early_mean_fighting": early_mean,
        "late_mean_fighting": late_mean,
        "absolute_drift": absolute_drift,
        "relative_drift": relative_drift,
        "fraction_zero_fighting": float(np.mean(fighting == 0)),
        "mean_fighting_fraction": mean_fighting_fraction,
        "mean_happy_fraction": optional_mean("happy_fraction"),
        "mean_entropy": optional_mean("spatial_entropy_local"),
        "mean_arrests": optional_mean("arrests_step"),
        "instability_score": float("nan"),
        "unstable_flag": False,
    }


def add_instability_score(records: list[dict[str, Any]], quantile: float) -> None:
    if not 0.0 < quantile < 1.0:
        raise ValueError("--instability-quantile must lie between 0 and 1")

    # Components are deliberately outcome-scale independent where possible.
    component_names = [
        "cv_fighting",
        "relative_drift",
        "max_step_jump",
        "mean_abs_step_jump",
    ]

    components = []
    for name in component_names:
        values = np.asarray([record[name] for record in records], dtype=float)
        components.append(robust_scale(values))

    # Equal weights: variation, drift, shock size, and general step-to-step noise.
    scores = np.mean(np.column_stack(components), axis=1)
    cutoff = float(np.quantile(scores, quantile))

    for record, score in zip(records, scores):
        record["instability_score"] = float(score)
        record["unstable_flag"] = bool(score >= cutoff)


def to_structured(records: list[dict[str, Any]]) -> np.ndarray:
    output = np.empty(len(records), dtype=SUMMARY_DTYPE)
    for index, record in enumerate(records):
        output[index] = tuple(record[name] for name in SUMMARY_DTYPE.names)
    return output


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_DTYPE.names))
        writer.writeheader()
        writer.writerows(records)


def save_3d_interactions(
    summary: np.ndarray,
    plots_dir: Path,
) -> None:
    z = summary["mean_fighting"]
    colour = summary["instability_score"]

    for x_name, y_name in itertools.combinations(PARAMETER_NAMES, 2):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            summary[x_name],
            summary[y_name],
            z,
            c=colour,
            cmap="viridis",
            s=38,
            alpha=0.85,
        )

        unstable = summary["unstable_flag"]
        if np.any(unstable):
            ax.scatter(
                summary[x_name][unstable],
                summary[y_name][unstable],
                z[unstable],
                facecolors="none",
                edgecolors="red",
                s=90,
                linewidths=1.2,
                label="Top unstable runs",
            )
            ax.legend(loc="best")

        ax.set_xlabel(PARAMETER_LABELS[x_name])
        ax.set_ylabel(PARAMETER_LABELS[y_name])
        ax.set_zlabel("Mean fighting fans")
        ax.set_title(
            f"Interaction: {PARAMETER_LABELS[x_name]} × "
            f"{PARAMETER_LABELS[y_name]}"
        )

        colorbar = fig.colorbar(scatter, ax=ax, pad=0.12, shrink=0.75)
        colorbar.set_label("Instability score")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"3d_{x_name}_x_{y_name}_mean_fighting.png",
            dpi=190,
        )
        plt.close(fig)


def save_parameter_instability_plots(
    summary: np.ndarray,
    plots_dir: Path,
) -> None:
    for parameter in PARAMETER_NAMES:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        scatter = ax.scatter(
            summary[parameter],
            summary["instability_score"],
            c=summary["mean_fighting"],
            cmap="plasma",
            alpha=0.8,
            s=42,
        )

        unstable = summary["unstable_flag"]
        if np.any(unstable):
            ax.scatter(
                summary[parameter][unstable],
                summary["instability_score"][unstable],
                facecolors="none",
                edgecolors="red",
                s=85,
                linewidths=1.2,
            )

        ax.set_xlabel(PARAMETER_LABELS[parameter])
        ax.set_ylabel("Instability score")
        ax.set_title(
            f"{PARAMETER_LABELS[parameter]} versus fighting instability"
        )
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Mean fighting fans")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"instability_vs_{parameter}.png",
            dpi=190,
        )
        plt.close(fig)


def save_diagnostic_scatter(
    summary: np.ndarray,
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        summary["mean_fighting"],
        summary["std_fighting"],
        c=summary["instability_score"],
        cmap="viridis",
        alpha=0.85,
        s=46,
    )
    unstable = summary["unstable_flag"]
    if np.any(unstable):
        ax.scatter(
            summary["mean_fighting"][unstable],
            summary["std_fighting"][unstable],
            facecolors="none",
            edgecolors="red",
            s=90,
            linewidths=1.2,
        )
    ax.set_xlabel("Mean fighting fans")
    ax.set_ylabel("SD fighting fans")
    ax.set_title("Average fighting versus temporal variation")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Instability score")
    fig.tight_layout()
    fig.savefig(plots_dir / "fighting_mean_vs_std.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        summary["mean_fighting"],
        summary["peak_fighting"],
        c=summary["relative_drift"],
        cmap="cividis",
        alpha=0.85,
        s=46,
    )
    ax.plot(
        [0, max(1.0, float(np.max(summary["peak_fighting"])))],
        [0, max(1.0, float(np.max(summary["peak_fighting"])))],
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlabel("Mean fighting fans")
    ax.set_ylabel("Peak fighting fans")
    ax.set_title("Peak fighting compared with average fighting")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Relative early–late drift")
    fig.tight_layout()
    fig.savefig(plots_dir / "fighting_mean_vs_peak.png", dpi=190)
    plt.close(fig)


def save_top_unstable_time_series(
    summary: np.ndarray,
    runs_dir: Path,
    plots_dir: Path,
    top_n: int,
) -> None:
    if top_n <= 0 or summary.size == 0:
        return

    order = np.argsort(summary["instability_score"])[::-1]
    selected = summary[order[: min(top_n, summary.size)]]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for row in selected:
        run_id = int(row["sample_id"])
        run = np.load(
            runs_dir / f"run_{run_id:04d}.npy",
            allow_pickle=False,
        )
        steps = (
            np.asarray(run["step"], dtype=int)
            if "step" in (run.dtype.names or ())
            else np.arange(len(run))
        )
        ax.plot(
            steps,
            run["fighting"],
            linewidth=1.3,
            alpha=0.82,
            label=f"{run_id} (score={row['instability_score']:.2f})",
        )

    ax.set_xlabel("Measurement step")
    ax.set_ylabel("Fighting fans")
    ax.set_title(f"Fighting trajectories of the {len(selected)} most unstable runs")
    ax.legend(
        title="Run ID",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(plots_dir / "top_unstable_fighting_timeseries.png", dpi=190)
    plt.close(fig)


def save_instability_histogram(
    summary: np.ndarray,
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(summary["instability_score"], bins=20)
    flagged = summary["instability_score"][summary["unstable_flag"]]
    if flagged.size:
        ax.axvline(
            float(np.min(flagged)),
            linestyle="--",
            linewidth=1.4,
            label="Unstable flag cutoff",
        )
        ax.legend()
    ax.set_xlabel("Instability score")
    ax.set_ylabel("Number of runs")
    ax.set_title("Distribution of fighting instability")
    fig.tight_layout()
    fig.savefig(plots_dir / "instability_distribution.png", dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_dir = args.data_dir / "runs"
    samples_path = args.data_dir / "sobol_samples.npy"

    if not runs_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {runs_dir}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Sample file not found: {samples_path}")

    samples = np.load(samples_path, allow_pickle=False)
    if samples.ndim != 2 or samples.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(
            f"Expected sobol_samples.npy shape (n, 4), found {samples.shape}"
        )

    run_paths = sorted(runs_dir.glob("run_*.npy"))
    if not run_paths:
        raise RuntimeError(f"No run_XXXX.npy files found in {runs_dir}")

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for path in run_paths:
        try:
            sample_id = run_id_from_path(path)
            if sample_id < 0 or sample_id >= len(samples):
                raise IndexError(
                    f"run ID {sample_id} outside sample matrix with {len(samples)} rows"
                )
            run = np.load(path, allow_pickle=False)
            records.append(
                calculate_raw_metrics(run, sample_id, samples[sample_id])
            )
        except Exception as exc:
            skipped.append({"file": str(path), "error": str(exc)})

    if not records:
        raise RuntimeError("No valid per-run files could be analyzed")

    records.sort(key=lambda item: item["sample_id"])
    add_instability_score(records, args.instability_quantile)
    summary = to_structured(records)

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    output_npy = args.data_dir / "plot_v2_run_diagnostics.npy"
    output_csv = args.data_dir / "plot_v2_run_diagnostics.csv"
    output_json = args.data_dir / "plot_v2_summary.json"

    np.save(output_npy, summary, allow_pickle=False)
    write_csv(output_csv, records)

    save_3d_interactions(summary, args.plots_dir)
    save_parameter_instability_plots(summary, args.plots_dir)
    save_diagnostic_scatter(summary, args.plots_dir)
    save_top_unstable_time_series(
        summary,
        runs_dir,
        args.plots_dir,
        args.top_unstable,
    )
    save_instability_histogram(summary, args.plots_dir)

    unstable_rows = summary[summary["unstable_flag"]]
    report = {
        "sample_rows": int(len(samples)),
        "available_run_files": int(len(run_paths)),
        "analyzed_runs": int(len(summary)),
        "skipped_runs": skipped,
        "instability_quantile": float(args.instability_quantile),
        "unstable_count": int(len(unstable_rows)),
        "unstable_run_ids": unstable_rows["sample_id"].astype(int).tolist(),
        "most_unstable_runs": [
            {
                "sample_id": int(row["sample_id"]),
                "instability_score": float(row["instability_score"]),
                "mean_fighting": float(row["mean_fighting"]),
                "std_fighting": float(row["std_fighting"]),
                "peak_fighting": float(row["peak_fighting"]),
                "relative_drift": float(row["relative_drift"]),
                **{
                    name: float(row[name])
                    for name in PARAMETER_NAMES
                },
            }
            for row in summary[
                np.argsort(summary["instability_score"])[::-1][
                    : min(args.top_unstable, len(summary))
                ]
            ]
        ],
    }
    output_json.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nPlot V2 complete")
    print("----------------")
    print(f"Sample rows:          {len(samples)}")
    print(f"Run files found:      {len(run_paths)}")
    print(f"Runs analyzed:        {len(summary)}")
    print(f"Runs skipped:         {len(skipped)}")
    print(f"Flagged unstable:     {len(unstable_rows)}")
    print(f"Diagnostics CSV:      {output_csv.resolve()}")
    print(f"Diagnostics NPY:      {output_npy.resolve()}")
    print(f"Summary JSON:         {output_json.resolve()}")
    print(f"Plots directory:      {args.plots_dir.resolve()}")


if __name__ == "__main__":
    main()
