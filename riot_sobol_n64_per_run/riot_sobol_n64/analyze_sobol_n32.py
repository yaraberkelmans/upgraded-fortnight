"""Analyze an N=32 Sobol run with four parameters (192 model runs).

This script expects the same folder structure and run_results.npy schema as the
N=64 version. With four parameters and calc_second_order=False:

    total runs = N * (D + 2) = 32 * (4 + 2) = 192

Usage
-----
python analyze_sobol_n32.py
python analyze_sobol_n32.py --data-dir data --plots-dir plots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol

from run_sobol import PROBLEM


BASE_SAMPLE_SIZE = 32
EXPECTED_RUNS = BASE_SAMPLE_SIZE * (PROBLEM["num_vars"] + 2)

OUTPUT_LABELS = {
    "mean_fighting": "Mean fighting fans",
    "std_fighting": "SD fighting fans",
    "peak_fighting": "Peak fighting fans",
    "mean_fighting_fraction": "Mean fighting fraction",
    "arrests_measurement": "Arrests in measurement period",
    "mean_arrests_per_step": "Mean arrests per step",
    "mean_spatial_entropy_local": "Mean local spatial entropy",
    "mean_similarity": "Mean similarity",
    "mean_happy_fraction": "Mean happy fraction",
    "mean_win_probability": "Mean perceived win probability",
    "mean_arrest_probability": "Mean perceived arrest probability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze first-order Sobol indices for the N=32 run."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.data_dir / "run_results.npy"
    if not results_path.exists():
        raise FileNotFoundError(f"Could not find {results_path}")

    results = np.load(results_path, allow_pickle=False)

    if results.shape[0] != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} rows for N={BASE_SAMPLE_SIZE} and "
            f"D={PROBLEM['num_vars']}, but found {results.shape[0]}."
        )

    if "valid" not in results.dtype.names:
        raise RuntimeError("run_results.npy has no 'valid' field.")

    if not np.all(results["valid"]):
        failed = results[~results["valid"]]

        if "failure_code" in results.dtype.names:
            codes, counts = np.unique(
                failed["failure_code"],
                return_counts=True,
            )
            failure_summary = dict(
                zip(codes.tolist(), counts.tolist())
            )
        else:
            failure_summary = {
                "invalid": int(len(failed)),
            }

        raise RuntimeError(
            "Cannot perform Sobol analysis with invalid runs: "
            f"{failure_summary}"
        )

    rows: list[dict[str, object]] = []
    json_out: dict[str, object] = {}

    available_outputs = [
        field
        for field in OUTPUT_LABELS
        if field in results.dtype.names
    ]

    missing_outputs = sorted(
        set(OUTPUT_LABELS) - set(available_outputs)
    )
    if missing_outputs:
        print(
            "Skipping fields not present in run_results.npy: "
            + ", ".join(missing_outputs)
        )

    for field in available_outputs:
        label = OUTPUT_LABELS[field]
        y = np.asarray(results[field], dtype=float)

        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"Non-finite values in output {field}")

        if np.var(y) == 0:
            print(f"Skipping {field}: zero variance")
            continue

        si = sobol.analyze(
            PROBLEM,
            y,
            calc_second_order=False,
            print_to_console=False,
            seed=2026,
        )

        s1 = np.asarray(si["S1"], dtype=float)
        conf = np.asarray(si["S1_conf"], dtype=float)

        json_out[field] = {
            "label": label,
            "base_sample_size": BASE_SAMPLE_SIZE,
            "total_runs": EXPECTED_RUNS,
            "S1": dict(zip(PROBLEM["names"], s1.tolist())),
            "S1_conf": dict(
                zip(PROBLEM["names"], conf.tolist())
            ),
        }

        for parameter, value, ci in zip(
            PROBLEM["names"],
            s1,
            conf,
        ):
            rows.append(
                {
                    "output": field,
                    "parameter": parameter,
                    "S1": float(value),
                    "S1_conf": float(ci),
                }
            )

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(PROBLEM["names"]))

        ax.bar(x, s1, yerr=conf, capsize=4)
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(
            x,
            PROBLEM["names"],
            rotation=20,
            ha="right",
        )
        ax.set_ylabel("First-order Sobol index (S1)")
        ax.set_title(f"{label} — N={BASE_SAMPLE_SIZE}")

        finite_lower = s1 - conf
        if np.any(np.isfinite(finite_lower)):
            ax.set_ylim(
                bottom=min(
                    -0.1,
                    float(np.nanmin(finite_lower)) - 0.05,
                )
            )

        fig.tight_layout()
        fig.savefig(
            args.plots_dir / f"sobol_N32_S1_{field}.png",
            dpi=180,
        )
        plt.close(fig)

    csv_path = args.data_dir / "sobol_N32_first_order.csv"
    json_path = args.data_dir / "sobol_N32_first_order.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output",
                "parameter",
                "S1",
                "S1_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(json_out, indent=2),
        encoding="utf-8",
    )

    if json_out:
        fields = list(json_out)
        matrix = np.array(
            [
                [
                    json_out[field]["S1"][parameter]
                    for parameter in PROBLEM["names"]
                ]
                for field in fields
            ],
            dtype=float,
        )

        fig, ax = plt.subplots(
            figsize=(9, max(5, 0.45 * len(fields)))
        )
        image = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(
            np.arange(len(PROBLEM["names"])),
            PROBLEM["names"],
            rotation=20,
            ha="right",
        )
        ax.set_yticks(
            np.arange(len(fields)),
            [OUTPUT_LABELS[field] for field in fields],
        )
        ax.set_title(
            f"First-order Sobol indices overview — N={BASE_SAMPLE_SIZE}"
        )
        fig.colorbar(image, ax=ax, label="S1")
        fig.tight_layout()
        fig.savefig(
            args.plots_dir / "sobol_N32_S1_overview.png",
            dpi=180,
        )
        plt.close(fig)

    print(f"Base sample size: {BASE_SAMPLE_SIZE}")
    print(f"Expected model runs: {EXPECTED_RUNS}")
    print(f"Sobol table: {csv_path.resolve()}")
    print(f"Sobol JSON: {json_path.resolve()}")
    print(f"Plots: {args.plots_dir.resolve()}")


if __name__ == "__main__":
    main()
