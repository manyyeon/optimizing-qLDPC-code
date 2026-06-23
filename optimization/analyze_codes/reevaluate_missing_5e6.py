#!/usr/bin/env python3
"""Reevaluate selected repeated-run codes with 5e6 Monte Carlo trials.

This script reproduces the reevaluation storage format already present in
the completed repeated-run HDF5 files.

Selection policy
----------------
For each HDF5 run:
  1. consider precision-evaluated rows when available;
  2. maximize HGP distance;
  3. minimize stored precision LER;
  4. minimize stored precision LER standard error;
  5. minimize weighted score;
  6. choose the smaller row index.

Saved HDF5 data
---------------
Run-group attributes:
    reeval_5000000_budget
    reeval_5000000_completed_runs
    reeval_5000000_early_stopped
    reeval_5000000_failures
    reeval_5000000_host
    reeval_5000000_ler
    reeval_5000000_ler_std
    reeval_5000000_pid
    reeval_5000000_row_idx
    reeval_5000000_runtime_seconds
    reeval_5000000_selection_policy
    reeval_5000000_started_utc
    reeval_5000000_state_sha256
    reeval_5000000_status
    reeval_5000000_timestamp_utc

Per-row datasets:
    reeval_5000000_selected
    reeval_5000000_ler
    reeval_5000000_ler_std
    reeval_5000000_failures
    reeval_5000000_completed_runs
    reeval_5000000_runtime_seconds

Historical record:
    reevaluations/budget_5000000/<UTC timestamp>/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from optimization.logical_guided_search.logical_guided_eval import (
    evaluate_mc,
    get_code_parameters_and_matrices,
)


SELECTION_POLICY = "distance_then_ler"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def infer_file_metadata(path: Path) -> tuple[str, int | None, int | None]:
    name = path.name.lower()

    if "greedy" in name:
        method = "greedy"
    elif "beam" in name:
        method = "beam"
    else:
        method = "unknown"

    c_match = re.search(r"_c(\d+)(?:_|\.|$)", name)
    run_match = re.search(r"_run(\d+)(?:_|\.|$)", name)

    code_index = int(c_match.group(1)) if c_match else None
    run_number = int(run_match.group(1)) if run_match else None
    return method, code_index, run_number


def find_run_group(h5_file: h5py.File) -> h5py.Group:
    matches: list[str] = []

    def visitor(_: str, obj: h5py.Group | h5py.Dataset) -> None:
        if (
            isinstance(obj, h5py.Group)
            and "states" in obj
            and (
                "distances_quantum" in obj
                or "score_d_q" in obj
            )
        ):
            matches.append(obj.name)

    h5_file.visititems(visitor)

    if not matches:
        raise ValueError("No run group containing states was found.")
    if len(matches) > 1:
        raise ValueError(
            f"Found {len(matches)} candidate run groups: {matches}"
        )

    return h5_file[matches[0]]


def numeric_dataset(
    group: h5py.Group,
    names: list[str],
    *,
    default: float = np.nan,
) -> np.ndarray:
    n_rows = group["states"].shape[0]

    for name in names:
        if name in group and isinstance(group[name], h5py.Dataset):
            return np.asarray(group[name][:], dtype=float)

    return np.full(n_rows, default, dtype=float)


def choose_row(group: h5py.Group) -> dict[str, Any]:
    n_rows = group["states"].shape[0]
    row_indices = np.arange(n_rows, dtype=int)

    distance = numeric_dataset(
        group,
        ["distances_quantum", "score_d_q"],
    )
    precision_ler = numeric_dataset(
        group,
        [
            "prec_logical_error_rates",
            "logical_error_rates",
            "screen_logical_error_rates",
        ],
    )
    precision_std = numeric_dataset(
        group,
        [
            "prec_logical_error_rates_std",
            "logical_error_rates_std",
            "screen_logical_error_rates_std",
        ],
    )
    score = numeric_dataset(
        group,
        ["low_weight_scores"],
        default=np.inf,
    )

    valid = np.isfinite(distance) & np.isfinite(precision_ler)

    # Prefer the rows that received the precision-budget evaluation.
    if "precision_selected" in group:
        precision_selected = np.asarray(
            group["precision_selected"][:],
            dtype=bool,
        )
        precision_valid = valid & precision_selected

        if np.any(precision_valid):
            valid = precision_valid

    valid_rows = row_indices[valid]
    if valid_rows.size == 0:
        raise ValueError(
            "No row has both a finite distance and stored LER."
        )

    # np.lexsort uses the final key as the primary key.
    order = np.lexsort(
        (
            valid_rows,
            score[valid_rows],
            precision_std[valid_rows],
            precision_ler[valid_rows],
            -distance[valid_rows],
        )
    )
    row_idx = int(valid_rows[order[0]])

    return {
        "row_idx": row_idx,
        "distance": float(distance[row_idx]),
        "stored_ler": float(precision_ler[row_idx]),
        "stored_ler_std": float(precision_std[row_idx]),
        "stored_score": float(score[row_idx]),
    }


def state_sha256(state: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(state)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def ensure_dataset(
    group: h5py.Group,
    name: str,
    *,
    length: int,
    dtype: Any,
    fill_value: Any,
) -> h5py.Dataset:
    if name in group:
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"{group.name}/{name} is not a dataset.")
        if dataset.shape != (length,):
            raise ValueError(
                f"{group.name}/{name} has shape {dataset.shape}; "
                f"expected {(length,)}."
            )
        return dataset

    data = np.full(length, fill_value, dtype=dtype)
    return group.create_dataset(name, data=data, dtype=dtype)


def update_selected_best(
    group: h5py.Group,
    *,
    row_idx: int,
    state: np.ndarray,
    selected: dict[str, Any],
) -> None:
    n_rows = group["states"].shape[0]

    final_best = ensure_dataset(
        group,
        "final_best",
        length=n_rows,
        dtype=np.uint8,
        fill_value=0,
    )
    final_best[:] = 0
    final_best[row_idx] = 1

    if "best_state" in group:
        best_state = group["best_state"]
        expected_shape = (1, state.size)
        if best_state.shape != expected_shape:
            del group["best_state"]
            best_state = group.create_dataset(
                "best_state",
                data=state.reshape(1, -1),
                dtype=state.dtype,
            )
        else:
            best_state[0, :] = state
    else:
        group.create_dataset(
            "best_state",
            data=state.reshape(1, -1),
            dtype=state.dtype,
        )

    group.attrs["best_row_idx"] = np.int64(row_idx)
    group.attrs["best_dist"] = np.int64(round(selected["distance"]))
    group.attrs["best_ler"] = np.float64(selected["stored_ler"])
    group.attrs["best_ler_std"] = np.float64(
        selected["stored_ler_std"]
    )
    group.attrs["best_score"] = np.float64(selected["stored_score"])
    group.attrs["min_cost"] = np.float64(selected["stored_ler"])
    group.attrs["selection_priority"] = (
        "distance_then_ler_then_score"
    )


def write_running_metadata(
    path: Path,
    *,
    prefix: str,
    timestamp: str,
    host: str,
    pid: int,
    force: bool,
) -> tuple[str, dict[str, Any], np.ndarray, float]:
    with h5py.File(path, "r+") as h5_file:
        group = find_run_group(h5_file)

        existing_status = str(
            decode_attr(group.attrs.get(f"{prefix}_status", ""))
        ).lower()

        if existing_status == "complete" and not force:
            raise RuntimeError(
                "A completed 5e6 reevaluation already exists. "
                "Use --force only when you intentionally want to replace it."
            )

        selected = choose_row(group)
        row_idx = selected["row_idx"]
        state = np.asarray(group["states"][row_idx]).copy()
        p = float(group.attrs.get("p", 0.03))
        sha256 = state_sha256(state)

        update_selected_best(
            group,
            row_idx=row_idx,
            state=state,
            selected=selected,
        )

        group.attrs[f"{prefix}_budget"] = np.int64(5_000_000)
        group.attrs[f"{prefix}_host"] = host
        group.attrs[f"{prefix}_pid"] = np.int64(pid)
        group.attrs[f"{prefix}_row_idx"] = np.int64(row_idx)
        group.attrs[f"{prefix}_selection_policy"] = SELECTION_POLICY
        group.attrs[f"{prefix}_started_utc"] = timestamp
        group.attrs[f"{prefix}_timestamp_utc"] = timestamp
        group.attrs[f"{prefix}_state_sha256"] = sha256
        group.attrs[f"{prefix}_status"] = "running"

        h5_file.flush()

        metadata = {
            **selected,
            "group_name": group.name,
            "p": p,
            "sha256": sha256,
        }

    return metadata["group_name"], metadata, state, p


def save_completed_result(
    path: Path,
    *,
    group_name: str,
    prefix: str,
    timestamp: str,
    state: np.ndarray,
    metadata: dict[str, Any],
    result: dict[str, Any],
) -> None:
    row_idx = int(metadata["row_idx"])
    host = socket.gethostname()
    runtime_seconds = float(result["runtime"])
    ler = float(result["ler"])
    ler_std = float(result["stderr"])
    failures = int(result["failures"])
    completed_runs = int(result["completed_runs"])
    early_stopped = bool(result["early_stopped"])

    with h5py.File(path, "r+") as h5_file:
        group = h5_file[group_name]
        n_rows = group["states"].shape[0]

        selected_ds = ensure_dataset(
            group,
            f"{prefix}_selected",
            length=n_rows,
            dtype=np.bool_,
            fill_value=False,
        )
        ler_ds = ensure_dataset(
            group,
            f"{prefix}_ler",
            length=n_rows,
            dtype=np.float64,
            fill_value=np.nan,
        )
        std_ds = ensure_dataset(
            group,
            f"{prefix}_ler_std",
            length=n_rows,
            dtype=np.float64,
            fill_value=np.nan,
        )
        failures_ds = ensure_dataset(
            group,
            f"{prefix}_failures",
            length=n_rows,
            dtype=np.int64,
            fill_value=-1,
        )
        completed_ds = ensure_dataset(
            group,
            f"{prefix}_completed_runs",
            length=n_rows,
            dtype=np.int64,
            fill_value=-1,
        )
        runtime_ds = ensure_dataset(
            group,
            f"{prefix}_runtime_seconds",
            length=n_rows,
            dtype=np.float64,
            fill_value=np.nan,
        )

        # Reset this reevaluation prefix before writing the selected row.
        selected_ds[:] = False
        ler_ds[:] = np.nan
        std_ds[:] = np.nan
        failures_ds[:] = -1
        completed_ds[:] = -1
        runtime_ds[:] = np.nan

        selected_ds[row_idx] = True
        ler_ds[row_idx] = ler
        std_ds[row_idx] = ler_std
        failures_ds[row_idx] = failures
        completed_ds[row_idx] = completed_runs
        runtime_ds[row_idx] = runtime_seconds

        group.attrs[f"{prefix}_budget"] = np.int64(5_000_000)
        group.attrs[f"{prefix}_completed_runs"] = np.int64(
            completed_runs
        )
        group.attrs[f"{prefix}_early_stopped"] = early_stopped
        group.attrs[f"{prefix}_failures"] = np.int64(failures)
        group.attrs[f"{prefix}_host"] = host
        group.attrs[f"{prefix}_ler"] = np.float64(ler)
        group.attrs[f"{prefix}_ler_std"] = np.float64(ler_std)
        group.attrs[f"{prefix}_pid"] = np.int64(os.getpid())
        group.attrs[f"{prefix}_row_idx"] = np.int64(row_idx)
        group.attrs[f"{prefix}_runtime_seconds"] = np.float64(
            runtime_seconds
        )
        group.attrs[f"{prefix}_selection_policy"] = SELECTION_POLICY
        group.attrs[f"{prefix}_started_utc"] = timestamp
        group.attrs[f"{prefix}_state_sha256"] = metadata["sha256"]
        group.attrs[f"{prefix}_status"] = "complete"
        group.attrs[f"{prefix}_timestamp_utc"] = timestamp

        # Preserve a history entry in the same format as completed files.
        history_root = group.require_group("reevaluations")
        budget_root = history_root.require_group("budget_5000000")

        history_name = timestamp
        suffix = 1
        while history_name in budget_root:
            history_name = f"{timestamp}_{suffix}"
            suffix += 1

        history = budget_root.create_group(history_name)
        history.attrs["budget"] = np.int64(5_000_000)
        history.attrs["completed_runs"] = np.int64(completed_runs)
        history.attrs["early_stopped"] = early_stopped
        history.attrs["failures"] = np.int64(failures)
        history.attrs["host"] = host
        history.attrs["ler"] = np.float64(ler)
        history.attrs["ler_std"] = np.float64(ler_std)
        history.attrs["row_idx"] = np.int64(row_idx)
        history.attrs["runtime_seconds"] = np.float64(
            runtime_seconds
        )
        history.attrs["selection_policy"] = SELECTION_POLICY
        history.attrs["state_sha256"] = metadata["sha256"]
        history.create_dataset(
            "state",
            data=state.reshape(1, -1),
            dtype=state.dtype,
        )

        h5_file.flush()


def save_failed_status(
    path: Path,
    *,
    group_name: str,
    prefix: str,
    error: Exception,
) -> None:
    try:
        with h5py.File(path, "r+") as h5_file:
            group = h5_file[group_name]
            group.attrs[f"{prefix}_status"] = "failed"
            group.attrs[f"{prefix}_error"] = repr(error)
            h5_file.flush()
    except Exception:
        pass


def reevaluate_file(
    path: Path,
    *,
    budget: int,
    workers: int,
    batch_size: int,
    force: bool,
    dry_run: bool,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    method, code_index, run_number = infer_file_metadata(path)
    prefix = f"reeval_{budget}"
    timestamp = utc_timestamp()
    host = socket.gethostname()
    pid = os.getpid()

    group_name, metadata, state, p = write_running_metadata(
        path,
        prefix=prefix,
        timestamp=timestamp,
        host=host,
        pid=pid,
        force=force,
    )

    print("=" * 72)
    print(f"{budget:,}-TRIAL BEST-CODE RE-EVALUATION")
    print(f"File: {path}")
    print(f"Run group: {group_name}")
    print(f"Method: {method}")
    print(f"C: {code_index}")
    print(f"Run: {run_number}")
    print(f"Selection policy: {SELECTION_POLICY}")
    print(f"Selected HDF5 row: {metadata['row_idx']}")
    print(f"Distance: {metadata['distance']}")
    print(
        "Stored precision LER: "
        f"{metadata['stored_ler']} ± {metadata['stored_ler_std']}"
    )
    print(f"Stored score: {metadata['stored_score']}")
    print(f"Physical error rate p: {p}")
    print(f"Budget: {budget}")
    print(f"Workers: {workers}")
    print(f"Batch size: {batch_size}")
    print(f"State SHA256: {metadata['sha256']}")
    print("=" * 72)

    if dry_run:
        print("DRY RUN: selection completed; Monte Carlo was not run.")
        with h5py.File(path, "r+") as h5_file:
            group = h5_file[group_name]
            group.attrs[f"{prefix}_status"] = "dry_run"
            h5_file.flush()
        return

    wall_start = time.time()

    try:
        _, Hx, Hz = get_code_parameters_and_matrices(state)

        label = (
            f"{prefix}_{method}_C{code_index}_run{run_number}"
            f"_row{metadata['row_idx']}"
        )

        result = evaluate_mc(
            Hx,
            Hz,
            p,
            budget,
            run_label=label,
            workers=workers,
            batch_size=batch_size,
        )

        wall_runtime = time.time() - wall_start

        save_completed_result(
            path,
            group_name=group_name,
            prefix=prefix,
            timestamp=timestamp,
            state=state,
            metadata=metadata,
            result=result,
        )

        print("\n>>> RE-EVALUATION RESULT")
        print(f"Selected row: {metadata['row_idx']}")
        print(f"Distance: {metadata['distance']}")
        print(
            f"LER: {float(result['ler']):.8f} "
            f"± {float(result['stderr']):.8f}"
        )
        print(f"Failures: {int(result['failures'])}")
        print(f"Completed trials: {int(result['completed_runs'])}")
        print(f"Evaluator runtime: {float(result['runtime']):.2f}s")
        print(f"Wall runtime: {wall_runtime:.2f}s")
        print(f"Saved prefix: {prefix}")
        print(f"Saved to: {path}")
        print(f"Run group: {group_name}")
        print(
            f"COMPLETED: {method}, C={code_index}, "
            f"run={run_number}"
        )

    except Exception as exc:
        save_failed_status(
            path,
            group_name=group_name,
            prefix=prefix,
            error=exc,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        required=True,
        nargs="+",
        type=Path,
        help="One or more repeated-run HDF5 files.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5_000_000,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an already-complete reevaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and report the row without Monte Carlo evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.budget != 5_000_000:
        raise ValueError(
            "This script is intended to reproduce the "
            "reeval_5000000 save format; use --budget 5000000."
        )
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    failures: list[tuple[Path, Exception]] = []

    for path in args.file:
        try:
            reevaluate_file(
                path,
                budget=args.budget,
                workers=args.workers,
                batch_size=args.batch_size,
                force=args.force,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            failures.append((path, exc))
            print(f"\nFAILED: {path}")
            print(repr(exc))

    if failures:
        print("\nFailed files:")
        for path, exc in failures:
            print(f"  {path}: {exc!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
