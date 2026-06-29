"""2-D interpolated heatmaps of fighting fraction over Sobol parameter pairs.

Reads ``<data_dir>/seed_*/run_results.npy``, averages QoIs for identical
parameter sets across seeds, then interpolates the scattered Sobol sample onto
a regular grid (cubic, 150 x 150) to produce three pairwise heatmaps for
``mean_fighting_fraction``.

Figures (displayed interactively)
----------------------------------
- similarity_threshold vs police_density
- similarity_threshold vs fight_threshold
- similarity_threshold vs hawk_dove_C

Run from project root:
    python DATA_PROCESSING/surfaceplots.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

# ── Data loading and aggregation ──────────────────────────────────────────────────────────────


def load_and_aggregate_data(base_dir="data"):
    """Load run_results.npy from all seed folders and average outputs for identical parameter sets."""
    base_path = Path(base_dir)
    seed_dirs = list(base_path.glob("seed_*"))

    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found in {base_dir}")

    all_results = []
    for d in seed_dirs:
        res_file = d / "run_results.npy"
        if res_file.exists():
            data = np.load(res_file)
            # Filter out invalid runs before aggregating
            data = data[data["valid"] == True]
            all_results.append(data)

    if not all_results:
        raise ValueError("No valid run_results.npy files found.")

    combined = np.concatenate(all_results)

    param_cols = ["similarity_threshold", "fight_threshold", "hawk_dove_C", "police_density"]

    params_only = combined[param_cols]
    unique_params, inverse_indices = np.unique(params_only, return_inverse=True)

    aggregated_data = []
    for i in range(len(unique_params)):
        mask = inverse_indices == i
        mean_fighting = combined["mean_fighting_fraction"][mask].mean()
        mean_arrests = combined["mean_arrests_per_step"][mask].mean()

        row = list(unique_params[i]) + [mean_fighting, mean_arrests]
        aggregated_data.append(row)

    dtype = [
        ("similarity_threshold", "f8"),
        ("fight_threshold", "f8"),
        ("hawk_dove_C", "f8"),
        ("police_density", "f8"),
        ("mean_fighting_fraction", "f8"),
        ("mean_arrests_per_step", "f8"),
    ]
    return np.array([tuple(row) for row in aggregated_data], dtype=dtype)


# ── Heatmap plot function ─────────────────────────────────────────────────────────────────────


def plot_heatmap(data, x_var, y_var, z_var, title, grid_res=150):
    """Interpolate scattered Sobol data onto a regular grid and plot a 2-D heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))

    x = data[x_var]
    y = data[y_var]
    z = data[z_var]

    # Build the interpolation grid
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)

    # Cubic interpolation; clip to prevent mathematical overshooting (e.g. negative fractions)
    Z = griddata((x, y), z, (X, Y), method="cubic")
    Z = np.clip(Z, 0, 1)

    heatmap = ax.pcolormesh(X, Y, Z, cmap="viridis", shading="auto", vmin=0, vmax=1)

    # Overlay actual Sobol sample points so interpolation confidence is visible
    ax.scatter(x, y, color="white", edgecolor="black", s=10, alpha=0.5, label="Actual Sobol Samples")

    ax.set_xlabel(x_var.replace("_", " ").title())
    ax.set_ylabel(y_var.replace("_", " ").title())
    ax.set_title(title)

    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label(z_var.replace("_", " ").title())
    ax.legend(loc="best", fontsize="small")

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading and aggregating data...")
    data = load_and_aggregate_data("data")

    z_target = "mean_fighting_fraction"

    print(f"Generating heatmaps for {z_target}...")

    plot_heatmap(
        data,
        x_var="similarity_threshold",
        y_var="police_density",
        z_var=z_target,
        title="Fighting Fraction: Similarity vs Police Density",
    )
    plot_heatmap(
        data,
        x_var="similarity_threshold",
        y_var="fight_threshold",
        z_var=z_target,
        title="Fighting Fraction: Similarity vs Fight Threshold",
    )
    plot_heatmap(
        data,
        x_var="similarity_threshold",
        y_var="hawk_dove_C",
        z_var=z_target,
        title="Fighting Fraction: Similarity vs Hawk/Dove Cost",
    )
