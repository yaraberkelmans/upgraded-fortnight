"""Audit an existing Sobol run folder and attach parameter values to failures.

Expected layout
---------------
data/
├── runs/
│   ├── run_0000.npy
│   ├── run_0001.npy
│   └── ...
├── invalid_runs.npy          # optional
├── sobol_samples.npy         # required
└── run_results.npy           # optional

Usage
-----
python audit_sobol_runs.py \
    --data-dir data \
    --expected-runs 384

Outputs
-------
data/
├── audit_summary.json
├── completed_runs.npy
├── invalid_runs_with_params.npy
├── missing_runs.npy
├── missing_runs_with_params.npy
├── unexplained_missing_runs.npy
├── unexplained_missing_with_params.npy
├── invalid_runs_with_params.csv
├── missing_runs_with_params.csv
└── unexplained_missing_with_params.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PARAMETER_NAMES = [
    "similarity_threshold",
    "fight_threshold",
    "hawk_dove_C",
    "police_density",
]


def extract_run_id(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except (ValueError, IndexError):
        return None


def extract_invalid_rows(invalid_path: Path) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Return invalid IDs plus any metadata already stored in invalid_runs.npy."""
    if not invalid_path.exists():
        return [], {}

    arr = np.load(invalid_path, allow_pickle=False)
    metadata: dict[int, dict[str, Any]] = {}

    if arr.dtype.names:
        id_field = next(
            (
                field
                for field in ("sample_id", "run_id", "id", "index")
                if field in arr.dtype.names
            ),
            None,
        )
        if id_field is None:
            raise ValueError(
                f"{invalid_path} has fields {arr.dtype.names}, but no run ID field."
            )

        ids: list[int] = []
        for row in arr:
            run_id = int(row[id_field])
            ids.append(run_id)

            row_data: dict[str, Any] = {}
            for field in arr.dtype.names:
                value = row[field]
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                row_data[field] = value

            metadata[run_id] = row_data

        return sorted(set(ids)), metadata

    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        ids = sorted({int(x) for x in arr})
        return ids, {}

    raise ValueError(
        f"Could not infer invalid IDs from {invalid_path}; "
        f"shape={arr.shape}, dtype={arr.dtype}"
    )


def build_records(
    run_ids: list[int],
    samples: np.ndarray,
    parameter_names: list[str],
    *,
    category: str,
    invalid_metadata: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    invalid_metadata = invalid_metadata or {}

    for run_id in run_ids:
        if run_id < 0 or run_id >= len(samples):
            records.append(
                {
                    "sample_id": int(run_id),
                    "category": category,
                    "error": "sample_id_out_of_range",
                }
            )
            continue

        values = samples[run_id]
        record: dict[str, Any] = {
            "sample_id": int(run_id),
            "category": category,
        }

        for name, value in zip(parameter_names, values):
            record[name] = float(value)

        if run_id in invalid_metadata:
            for key, value in invalid_metadata[run_id].items():
                if key not in record:
                    record[key] = value

        records.append(record)

    return records


def records_to_structured_array(
    records: list[dict[str, Any]],
    parameter_names: list[str],
) -> np.ndarray:
    dtype = [
        ("sample_id", np.int64),
        ("category", "U32"),
    ]
    dtype.extend((name, np.float64) for name in parameter_names)

    # Preserve common failure metadata when present.
    optional_string_fields = ["reason", "failure_reason", "message", "status"]
    for field in optional_string_fields:
        if any(field in record for record in records):
            dtype.append((field, "U256"))

    optional_numeric_fields = [
        "warmup_steps",
        "runtime_seconds",
    ]
    for field in optional_numeric_fields:
        if any(field in record for record in records):
            dtype.append((field, np.float64))

    arr = np.zeros(len(records), dtype=dtype)

    for i, record in enumerate(records):
        arr["sample_id"][i] = int(record.get("sample_id", -1))
        arr["category"][i] = str(record.get("category", ""))

        for name in parameter_names:
            arr[name][i] = float(record.get(name, np.nan))

        for field in optional_string_fields:
            if field in arr.dtype.names:
                arr[field][i] = str(record.get(field, ""))

        for field in optional_numeric_fields:
            if field in arr.dtype.names:
                value = record.get(field, np.nan)
                try:
                    arr[field][i] = float(value)
                except (TypeError, ValueError):
                    arr[field][i] = np.nan

    return arr


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def audit_sobol_runs(
    data_dir: Path,
    expected_runs: int,
    parameter_names: list[str],
) -> dict[str, Any]:
    runs_dir = data_dir / "runs"
    samples_path = data_dir / "sobol_samples.npy"
    invalid_path = data_dir / "invalid_runs.npy"

    if not samples_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {samples_path}"
        )

    samples = np.load(samples_path, allow_pickle=False)

    if samples.ndim != 2:
        raise ValueError(
            f"{samples_path} should be a 2D array; got shape {samples.shape}"
        )

    if samples.shape[1] != len(parameter_names):
        raise ValueError(
            f"Sample matrix has {samples.shape[1]} columns, but "
            f"{len(parameter_names)} parameter names were supplied."
        )

    if expected_runs != len(samples):
        print(
            "Warning: expected_runs does not equal the number of rows in "
            f"sobol_samples.npy ({expected_runs} vs {len(samples)})."
        )

    completed_ids = sorted(
        run_id
        for path in runs_dir.glob("run_*.npy")
        if (run_id := extract_run_id(path)) is not None
    )

    invalid_ids, invalid_metadata = extract_invalid_rows(invalid_path)

    expected_ids = set(range(expected_runs))
    completed_set = set(completed_ids)
    invalid_set = set(invalid_ids)

    missing_ids = sorted(expected_ids - completed_set)
    unexplained_missing_ids = sorted(
        expected_ids - completed_set - invalid_set
    )
    invalid_without_run_ids = sorted(invalid_set - completed_set)
    invalid_with_run_ids = sorted(invalid_set & completed_set)
    out_of_range_ids = sorted(
        (completed_set | invalid_set) - expected_ids
    )

    invalid_records = build_records(
        invalid_ids,
        samples,
        parameter_names,
        category="invalid",
        invalid_metadata=invalid_metadata,
    )
    missing_records = build_records(
        missing_ids,
        samples,
        parameter_names,
        category="missing",
    )
    unexplained_records = build_records(
        unexplained_missing_ids,
        samples,
        parameter_names,
        category="unexplained_missing",
    )

    data_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        data_dir / "completed_runs.npy",
        np.asarray(completed_ids, dtype=np.int64),
    )
    np.save(
        data_dir / "missing_runs.npy",
        np.asarray(missing_ids, dtype=np.int64),
    )
    np.save(
        data_dir / "unexplained_missing_runs.npy",
        np.asarray(unexplained_missing_ids, dtype=np.int64),
    )

    np.save(
        data_dir / "invalid_runs_with_params.npy",
        records_to_structured_array(invalid_records, parameter_names),
    )
    np.save(
        data_dir / "missing_runs_with_params.npy",
        records_to_structured_array(missing_records, parameter_names),
    )
    np.save(
        data_dir / "unexplained_missing_with_params.npy",
        records_to_structured_array(unexplained_records, parameter_names),
    )

    write_csv(
        data_dir / "invalid_runs_with_params.csv",
        invalid_records,
    )
    write_csv(
        data_dir / "missing_runs_with_params.csv",
        missing_records,
    )
    write_csv(
        data_dir / "unexplained_missing_with_params.csv",
        unexplained_records,
    )

    summary: dict[str, Any] = {
        "expected_runs": int(expected_runs),
        "sample_rows": int(len(samples)),
        "parameter_names": parameter_names,
        "completed_count": len(completed_ids),
        "invalid_count": len(invalid_ids),
        "missing_count": len(missing_ids),
        "unexplained_missing_count": len(unexplained_missing_ids),
        "invalid_without_run_count": len(invalid_without_run_ids),
        "invalid_with_run_count": len(invalid_with_run_ids),
        "completed_ids": completed_ids,
        "invalid_ids": invalid_ids,
        "missing_ids": missing_ids,
        "unexplained_missing_ids": unexplained_missing_ids,
        "invalid_without_run_ids": invalid_without_run_ids,
        "invalid_with_run_ids": invalid_with_run_ids,
        "out_of_range_ids": out_of_range_ids,
        "invalid_parameter_values": invalid_records,
        "unexplained_missing_parameter_values": unexplained_records,
    }

    with (data_dir / "audit_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit existing Sobol run files and attach sampled parameter "
            "values to invalid and missing run IDs."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--parameter-names",
        nargs="+",
        default=DEFAULT_PARAMETER_NAMES,
        help=(
            "Names in the exact same order as the columns in "
            "sobol_samples.npy."
        ),
    )
    args = parser.parse_args()

    summary = audit_sobol_runs(
        data_dir=args.data_dir,
        expected_runs=args.expected_runs,
        parameter_names=list(args.parameter_names),
    )

    print("\nSobol run audit")
    print("----------------")
    print(f"Expected runs:               {summary['expected_runs']}")
    print(f"Completed run files:         {summary['completed_count']}")
    print(f"Invalid runs recorded:       {summary['invalid_count']}")
    print(f"Missing run files:           {summary['missing_count']}")
    print(
        "Unexplained missing runs:   "
        f"{summary['unexplained_missing_count']}"
    )

    if summary["invalid_ids"]:
        print("\nInvalid run IDs:")
        print(summary["invalid_ids"])

    if summary["unexplained_missing_ids"]:
        print("\nUnexplained missing run IDs:")
        print(summary["unexplained_missing_ids"])

    print("\nCreated files:")
    for name in (
        "audit_summary.json",
        "completed_runs.npy",
        "invalid_runs_with_params.npy",
        "invalid_runs_with_params.csv",
        "missing_runs.npy",
        "missing_runs_with_params.npy",
        "missing_runs_with_params.csv",
        "unexplained_missing_runs.npy",
        "unexplained_missing_with_params.npy",
        "unexplained_missing_with_params.csv",
    ):
        print(f"- {args.data_dir / name}")


if __name__ == "__main__":
    main()
