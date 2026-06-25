"""Quick inspection and plotting for arrest events and entropy overviews.

Run after run_sobol.py:
    python analyze_events.py --data-dir data --plots-dir plots_events
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--plots-dir", type=Path, default=Path("plots_events"))
    return p.parse_args()


def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    overview_path = args.data_dir / "run_overview.npy"
    if overview_path.exists():
        overview = np.load(overview_path, allow_pickle=False)
        valid = overview[overview["valid"]]
        if len(valid):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(valid["warmup_steps"], valid["warmup_entropy"], alpha=0.7)
            ax.set_xlabel("Warm-up steps")
            ax.set_ylabel("Entropy at end of warm-up")
            ax.set_title("Warm-up duration versus final warm-up entropy")
            fig.tight_layout()
            fig.savefig(args.plots_dir / "warmup_steps_vs_entropy.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                valid["start_measurement_entropy"],
                valid["end_measurement_entropy"],
                alpha=0.7,
            )
            low = float(min(valid["start_measurement_entropy"].min(), valid["end_measurement_entropy"].min()))
            high = float(max(valid["start_measurement_entropy"].max(), valid["end_measurement_entropy"].max()))
            ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
            ax.set_xlabel("Entropy at start of measurement")
            ax.set_ylabel("Entropy at end of measurement")
            ax.set_title("Spatial entropy change during measurement")
            fig.tight_layout()
            fig.savefig(args.plots_dir / "entropy_start_vs_end.png", dpi=180)
            plt.close(fig)

    arrest_files = sorted((args.data_dir / "arrests").glob("arrests_*.npy"))
    event_arrays = [np.load(path, allow_pickle=False) for path in arrest_files]
    nonempty = [arr for arr in event_arrays if len(arr)]
    if nonempty:
        events = np.concatenate(nonempty)
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.arange(0, 102) - 0.5
        ax.hist(events["step"], bins=bins)
        ax.set_xlabel("Measurement step")
        ax.set_ylabel("Arrested fans")
        ax.set_title("Arrest timing during measurement")
        fig.tight_layout()
        fig.savefig(args.plots_dir / "arrests_by_step.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        original = events[~events["is_respawn"]]["aggressiveness"]
        respawn = events[events["is_respawn"]]["aggressiveness"]
        if len(original):
            ax.hist(original, bins=20, alpha=0.6, label="Original")
        if len(respawn):
            ax.hist(respawn, bins=20, alpha=0.6, label="Respawn")
        ax.set_xlabel("Aggressiveness")
        ax.set_ylabel("Arrested fans")
        ax.set_title("Aggressiveness of arrested fans")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plots_dir / "arrested_aggressiveness.png", dpi=180)
        plt.close(fig)

        print(f"Arrest events: {len(events)}")
    else:
        print("No arrest events found.")

    print(f"Plots saved to: {args.plots_dir.resolve()}")


if __name__ == "__main__":
    main()
