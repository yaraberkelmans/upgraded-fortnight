"""Compact NumPy recording for aggregate riot-model system state.

The recorder stores one row per measurement step in a structured NumPy array.
It is intentionally independent of Mesa's DataCollector, so Sobol runs can use
``collect_data=False`` while retaining only the outputs needed for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .fan import FanGroup
except ImportError:
    from fan import FanGroup


SYSTEM_STATE_DTYPE = np.dtype(
    [
        ("step", np.int32),
        ("phase", "U12"),
        ("in_warmup", np.bool_),
        ("n_fans", np.int32),
        ("home", np.int32),
        ("away", np.int32),
        ("police", np.int32),
        ("happy", np.int32),
        ("unhappy", np.int32),
        ("fighting", np.int32),
        ("moves", np.int32),
        ("arrests_step", np.int32),
        ("total_arrests", np.int32),
        ("avg_similarity", np.float64),
        ("spatial_entropy", np.float64),
        ("spatial_entropy_local", np.float64),
        ("entropy_cv_local", np.float64),
        ("avg_aggressiveness", np.float64),
        ("avg_win_probability", np.float64),
        ("avg_arrest_probability", np.float64),
        ("avg_move_distance", np.float64),
        ("avg_move_distance_moved", np.float64),
        ("fighting_fraction", np.float64),
        ("home_fraction_current", np.float64),
    ]
)


def capture_system_state(model, step: int, phase: str = "measurement") -> tuple:
    """Return one aggregate system-state row matching ``SYSTEM_STATE_DTYPE``.

    ``spatial_entropy`` uses ``zone_size`` and
    ``spatial_entropy_local`` uses ``zone_size_fine``.
    """
    n_fans = len(model.fans)
    home = int(model.count_group(FanGroup.HOME))
    away = int(model.count_group(FanGroup.AWAY))
    fighting = int(model.count_fighting_fans())

    # _zone_entropy is available in the current refactor and reuses cached
    # occupancy planes. Calling it here avoids requiring the DataCollector.
    spatial_entropy = float(
        model._zone_entropy(model.segregation_params.zone_size)
    )
    spatial_entropy_local = float(
        model._zone_entropy(model.segregation_params.zone_size_fine)
    )

    return (
        int(step),
        str(phase),
        bool(model.in_warmup),
        int(n_fans),
        home,
        away,
        int(model.count_police()),
        int(model.count_happy()),
        int(model.count_unhappy()),
        fighting,
        int(model.moves_this_step),
        int(model.arrests_this_step),
        int(model.total_arrests),
        float(model.average_similarity()),
        spatial_entropy,
        spatial_entropy_local,
        float(model.entropy_cv_fine()),
        float(model.average_aggressiveness()),
        float(model.average_perceived_win_probability()),
        float(model.average_perceived_arrest_probability()),
        float(model.average_last_move_distance()),
        float(model.average_last_move_distance_of_moved_fans()),
        fighting / n_fans if n_fans else 0.0,
        home / n_fans if n_fans else 0.0,
    )


@dataclass
class SystemStateRecorder:
    """Collect aggregate model rows and expose them as a NumPy array."""

    _rows: list[tuple] = field(default_factory=list)

    def record(self, model, step: int, phase: str = "measurement") -> None:
        self._rows.append(capture_system_state(model, step, phase))

    def extend(self, rows: Iterable[tuple]) -> None:
        self._rows.extend(rows)

    def to_array(self) -> np.ndarray:
        return np.asarray(self._rows, dtype=SYSTEM_STATE_DTYPE)

    def clear(self) -> None:
        self._rows.clear()

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        """Save the recorded rows to .npz (default) or .npy.

        A compressed ``.npz`` contains the structured array under the key
        ``system_state``. A ``.npy`` file stores the array directly.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = self.to_array()

        if path.suffix == ".npy":
            np.save(path, array, allow_pickle=False)
            return path

        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        saver = np.savez_compressed if compressed else np.savez
        saver(
            path,
            system_state=array,
            columns=np.asarray(array.dtype.names, dtype="U40"),
        )
        return path


def load_system_state(path: str | Path) -> np.ndarray:
    """Load a structured system-state array from .npy or .npz."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as data:
        return data["system_state"]


def summarize_measurement(array: np.ndarray) -> dict[str, float]:
    """Return common scalar outputs suitable for separate Sobol analyses."""
    if array.size == 0:
        raise ValueError("Cannot summarize an empty system-state array")

    measurement = array[array["phase"] == "measurement"]
    if measurement.size == 0:
        measurement = array

    return {
        "mean_fighting": float(np.mean(measurement["fighting"])),
        "std_fighting": float(np.std(measurement["fighting"])),
        "peak_fighting": float(np.max(measurement["fighting"])),
        "mean_fighting_fraction": float(
            np.mean(measurement["fighting_fraction"])
        ),
        "arrests_measurement": float(np.sum(measurement["arrests_step"])),
        "mean_spatial_entropy": float(
            np.mean(measurement["spatial_entropy"])
        ),
        "mean_spatial_entropy_local": float(
            np.mean(measurement["spatial_entropy_local"])
        ),
        "mean_similarity": float(np.mean(measurement["avg_similarity"])),
    }
