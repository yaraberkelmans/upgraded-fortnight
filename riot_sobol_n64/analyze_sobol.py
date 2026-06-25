"""Analyze the 384-run Sobol pilot and create first-order plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol

from run_sobol import PROBLEM

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    results = np.load(args.data_dir / "run_results.npy", allow_pickle=False)
    if results.shape[0] != 384:
        raise RuntimeError(f"Expected 384 rows, found {results.shape[0]}")
    if not np.all(results["valid"]):
        failed = results[~results["valid"]]
        codes, counts = np.unique(failed["failure_code"], return_counts=True)
        summary = dict(zip(codes.tolist(), counts.tolist()))
        raise RuntimeError(f"Cannot perform Sobol analysis with invalid runs: {summary}")

    rows = []
    json_out = {}
    for field, label in OUTPUT_LABELS.items():
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
            "S1": dict(zip(PROBLEM["names"], s1.tolist())),
            "S1_conf": dict(zip(PROBLEM["names"], conf.tolist())),
        }
        for parameter, value, ci in zip(PROBLEM["names"], s1, conf):
            rows.append({
                "output": field,
                "parameter": parameter,
                "S1": float(value),
                "S1_conf": float(ci),
            })

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(PROBLEM["names"]))
        ax.bar(x, s1, yerr=conf, capsize=4)
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(x, PROBLEM["names"], rotation=20, ha="right")
        ax.set_ylabel("First-order Sobol index (S1)")
        ax.set_title(label)
        ax.set_ylim(bottom=min(-0.1, float(np.nanmin(s1 - conf)) - 0.05))
        fig.tight_layout()
        fig.savefig(args.plots_dir / f"sobol_S1_{field}.png", dpi=180)
        plt.close(fig)

    with (args.data_dir / "sobol_first_order.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["output", "parameter", "S1", "S1_conf"])
        writer.writeheader()
        writer.writerows(rows)
    (args.data_dir / "sobol_first_order.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")

    # Overview heatmap of all S1 values.
    fields = list(json_out)
    matrix = np.array([[json_out[f]["S1"][p] for p in PROBLEM["names"]] for f in fields])
    fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(fields))))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(PROBLEM["names"])), PROBLEM["names"], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(fields)), [OUTPUT_LABELS[f] for f in fields])
    ax.set_title("First-order Sobol indices overview")
    fig.colorbar(image, ax=ax, label="S1")
    fig.tight_layout()
    fig.savefig(args.plots_dir / "sobol_S1_overview.png", dpi=180)
    plt.close(fig)

    print(f"Sobol tables: {(args.data_dir / 'sobol_first_order.csv').resolve()}")
    print(f"Plots: {args.plots_dir.resolve()}")


if __name__ == "__main__":
    main()
