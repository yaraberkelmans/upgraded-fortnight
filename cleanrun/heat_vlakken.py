"""Create heat surfaces for Sobol parameter pairs.

This script uses the diagnostics created by plot_v2.py:

    data/plot_v2_run_diagnostics.npy

For each pair of parameters it creates:
- a 2D filled heat/contour map of mean fighting;
- a 2D filled heat/contour map of instability;
- a 3D triangulated surface of mean fighting.

Important:
These are projected pairwise views. The other two parameters still vary across
the sampled points, so the surfaces show an exploratory interpolation rather
than a controlled two-parameter experiment.

Usage
-----
python heat_vlakken.py
python heat_vlakken.py --data-dir data --plots-dir plots_heat
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


PARAMETERS = [
    "similarity_threshold",
    "fight_threshold",
    "hawk_dove_C",
    "police_density",
]

LABELS = {
    "similarity_threshold": "Similarity threshold",
    "fight_threshold": "Fight threshold",
    "hawk_dove_C": "Hawk–dove C",
    "police_density": "Police density",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create heat maps and 3D surfaces from plot_v2 diagnostics."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots_heat"))
    parser.add_argument(
        "--levels",
        type=int,
        default=18,
        help="Number of contour levels.",
    )
    return parser.parse_args()


def valid_mask(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y) & np.isfinite(z)


def make_triangulation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[mtri.Triangulation, np.ndarray, np.ndarray, np.ndarray]:
    mask = valid_mask(x, y, z)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    z = np.asarray(z[mask], dtype=float)

    if len(x) < 3:
        raise ValueError("At least three finite points are required.")

    triangulation = mtri.Triangulation(x, y)

    # Mask extremely stretched triangles. This reduces interpolation bridges
    # across sparse parts of the Sobol projection.
    triangles = triangulation.triangles
    x_tri = x[triangles]
    y_tri = y[triangles]

    side_a = np.hypot(
        x_tri[:, 1] - x_tri[:, 0],
        y_tri[:, 1] - y_tri[:, 0],
    )
    side_b = np.hypot(
        x_tri[:, 2] - x_tri[:, 1],
        y_tri[:, 2] - y_tri[:, 1],
    )
    side_c = np.hypot(
        x_tri[:, 0] - x_tri[:, 2],
        y_tri[:, 0] - y_tri[:, 2],
    )
    longest = np.maximum.reduce([side_a, side_b, side_c])

    cutoff = np.quantile(longest, 0.95)
    triangulation.set_mask(longest > cutoff)

    return triangulation, x, y, z


def plot_heatmap(
    data: np.ndarray,
    x_name: str,
    y_name: str,
    z_name: str,
    z_label: str,
    plots_dir: Path,
    levels: int,
) -> None:
    triangulation, x, y, z = make_triangulation(
        data[x_name],
        data[y_name],
        data[z_name],
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    contour = ax.tricontourf(
        triangulation,
        z,
        levels=levels,
        cmap="viridis",
    )
    ax.scatter(
        x,
        y,
        c=z,
        cmap="viridis",
        s=16,
        edgecolors="black",
        linewidths=0.2,
        alpha=0.75,
    )

    ax.set_xlabel(LABELS[x_name])
    ax.set_ylabel(LABELS[y_name])
    ax.set_title(
        f"{z_label}: {LABELS[x_name]} × {LABELS[y_name]}"
    )

    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label(z_label)

    fig.tight_layout()
    fig.savefig(
        plots_dir / f"heat_{z_name}_{x_name}_x_{y_name}.png",
        dpi=200,
    )
    plt.close(fig)


def plot_3d_surface(
    data: np.ndarray,
    x_name: str,
    y_name: str,
    plots_dir: Path,
) -> None:
    triangulation, x, y, z = make_triangulation(
        data[x_name],
        data[y_name],
        data["mean_fighting"],
    )

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_trisurf(
        triangulation,
        z,
        cmap="viridis",
        linewidth=0.25,
        antialiased=True,
        alpha=0.88,
    )
    ax.scatter(
        x,
        y,
        z,
        c=z,
        cmap="viridis",
        s=13,
        alpha=0.7,
    )

    ax.set_xlabel(LABELS[x_name])
    ax.set_ylabel(LABELS[y_name])
    ax.set_zlabel("Mean fighting fans")
    ax.set_title(
        f"Fighting surface: {LABELS[x_name]} × {LABELS[y_name]}"
    )

    colorbar = fig.colorbar(
        surface,
        ax=ax,
        pad=0.12,
        shrink=0.72,
    )
    colorbar.set_label("Mean fighting fans")

    fig.tight_layout()
    fig.savefig(
        plots_dir / f"surface_mean_fighting_{x_name}_x_{y_name}.png",
        dpi=200,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    diagnostics_path = args.data_dir / "plot_v2_run_diagnostics.npy"

    if not diagnostics_path.exists():
        raise FileNotFoundError(
            f"Could not find {diagnostics_path}. Run plot_v2.py first."
        )

    data = np.load(diagnostics_path, allow_pickle=False)

    required = set(PARAMETERS) | {
        "mean_fighting",
        "instability_score",
    }
    missing = required - set(data.dtype.names or ())
    if missing:
        raise ValueError(
            f"Diagnostics file misses fields: {sorted(missing)}"
        )

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    for x_name, y_name in itertools.combinations(PARAMETERS, 2):
        plot_heatmap(
            data,
            x_name,
            y_name,
            "mean_fighting",
            "Mean fighting fans",
            args.plots_dir,
            args.levels,
        )
        plot_heatmap(
            data,
            x_name,
            y_name,
            "instability_score",
            "Instability score",
            args.plots_dir,
            args.levels,
        )
        plot_3d_surface(
            data,
            x_name,
            y_name,
            args.plots_dir,
        )

    print("\nHeat surfaces complete")
    print("----------------------")
    print(f"Runs used:       {len(data)}")
    print(f"Output folder:   {args.plots_dir.resolve()}")
    print("Created 18 plots:")
    print("- 6 fighting heat maps")
    print("- 6 instability heat maps")
    print("- 6 three-dimensional fighting surfaces")


if __name__ == "__main__":
    main()
