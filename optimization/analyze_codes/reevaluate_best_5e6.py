#!/usr/bin/env python3
"""
Re-evaluate the best code from one repeated qLDPC search run.

Default selection policy:
    1. Highest distance among precision-evaluated Phase-2 candidates
    2. Lowest stored precision LER
    3. Lowest low-weight score
    4. Lowest HDF5 row index

The 5e6 result is stored separately from the original 1e5 precision result.

Stored on the run group:
    attrs["reeval_5000000_row_idx"]
    attrs["reeval_5000000_ler"]
    attrs["reeval_5000000_ler_std"]
    attrs["reeval_5000000_runtime_seconds"]
    attrs["reeval_5000000_completed_runs"]
    attrs["reeval_5000000_failures"]
    attrs["reeval_5000000_timestamp"]
    attrs["reeval_5000000_status"]

Row-wise datasets:
    reeval_5000000_selected
    reeval_5000000_ler
    reeval_5000000_ler_std
    reeval_5000000_runtime_seconds
    reeval_5000000_completed_runs
    reeval_5000000_failures

A timestamped history entry is also written below:
    reevaluations/budget_5000000/<timestamp>/
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from optimization.experiments_settings import (
    codes,
    from_edgelist,
    noise_levels,
)
from optimization.logical_guided_search.logical_guided_eval import (
    evaluate_mc,
    get_code_parameters_and_matrices,
)


DISTANCE_NAMES = (
    "distances_quantum",
    "quantum_distance",
    "hgp_distance",
    "logical_distance",
    "distance",
    "distances",
    "d_q",
    "score_d_q",
)
LER_NAMES = (
    "prec_logical_error_rates",
    "logical_error_rates",
    "logical_error_rate",
    "ler",
)
LER_STD_NAMES = (
    "prec_logical_error_rates_std",
    "logical_error_rates_std",
    "logical_error_rate_std",
    "ler_std",
    "std",
)
SCORE_NAMES = (
    "low_weight_scores",
    "low_weight_score",
    "weight_score",
    "logical_score",
    "score",
    "scores",
)
PRECISION_SELECTED_NAMES = (
    "precision_selected",
    "selected_for_precision",
    "is_precision_selected",
)
FINAL_BEST_NAMES = (
    "final_best",
    "is_final_best",
)
STATE_NAMES = (
    "states",
    "edge_lists",
    "edges",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_run_groups(h5: h5py.File) -> list[str]:
    groups: list[str] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Group):
            base = name.rsplit("/", 1)[-1]
            if base.startswith("logical_guided"):
                groups.append("/" + name.lstrip("/"))

    h5.visititems(visitor)
    return groups


def choose_run_group(h5: h5py.File, requested: str | None) -> str:
    if requested is not None:
        if requested not in h5:
            raise KeyError(f"Run group not found: {requested}")
        return requested

    groups = find_run_groups(h5)
    if len(groups) != 1:
        raise RuntimeError(
            f"Expected exactly one logical-guided run group, found {groups}. "
            "Pass --run-group explicitly."
        )
    return groups[0]


def find_dataset(
    group: h5py.Group,
    names: Iterable[str],
    *,
    required: bool = True,
) -> h5py.Dataset | None:
    mapping: dict[str, list[h5py.Dataset]] = {}

    def visitor(_: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            basename = obj.name.rsplit("/", 1)[-1].lower()
            mapping.setdefault(basename, []).append(obj)

    group.visititems(visitor)

    for name in names:
        matches = mapping.get(name.lower(), [])
        if matches:
            return sorted(
                matches,
                key=lambda ds: (ds.name.count("/"), len(ds.name)),
            )[0]

    if required:
        raise KeyError(
            f"None of the datasets {tuple(names)} were found under {group.name}."
        )
    return None


def read_1d(
    group: h5py.Group,
    names: Iterable[str],
    *,
    required: bool = True,
    dtype=None,
) -> np.ndarray | None:
    dataset = find_dataset(group, names, required=required)
    if dataset is None:
        return None

    array = np.asarray(dataset[()])
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1 and int(np.prod(array.shape[1:])) == 1:
        array = array.reshape(array.shape[0])

    if array.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional dataset for {dataset.name}, "
            f"found shape {array.shape}."
        )

    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def state_hash(edge_list: np.ndarray) -> str:
    normalized = np.ascontiguousarray(edge_list, dtype=np.uint32)
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def saved_best_row(group: h5py.Group, n_rows: int) -> int:
    if "best_row_idx" in group.attrs:
        row = int(group.attrs["best_row_idx"])
        if 0 <= row < n_rows:
            return row

    final_best = read_1d(
        group,
        FINAL_BEST_NAMES,
        required=False,
    )
    if final_best is not None:
        rows = np.flatnonzero(final_best.astype(bool))
        if len(rows):
            return int(rows[-1])

    if "best_state" in group:
        states_ds = find_dataset(group, STATE_NAMES)
        best_state = np.asarray(group["best_state"][()])
        if best_state.ndim == 2 and best_state.shape[0] == 1:
            best_state = best_state[0]

        for row in range(states_ds.shape[0]):
            candidate = np.asarray(states_ds[row])
            if np.array_equal(candidate, best_state):
                return int(row)

    raise RuntimeError(
        "Could not determine the saved best row from best_row_idx, "
        "final_best, or best_state."
    )


def distance_then_ler_row(group: h5py.Group) -> int:
    distance = read_1d(group, DISTANCE_NAMES, dtype=float)
    ler = read_1d(group, LER_NAMES, dtype=float)
    score = read_1d(group, SCORE_NAMES, required=False, dtype=float)
    precision_selected = read_1d(
        group,
        PRECISION_SELECTED_NAMES,
        required=False,
    )

    n_rows = len(distance)
    if len(ler) != n_rows:
        raise ValueError(
            f"Distance has {n_rows} rows but LER has {len(ler)} rows."
        )

    if score is None:
        score = np.full(n_rows, np.inf)
    elif len(score) != n_rows:
        raise ValueError(
            f"Distance has {n_rows} rows but score has {len(score)} rows."
        )

    if precision_selected is None:
        precision_mask = np.isfinite(ler) & (ler >= 0)
    else:
        if len(precision_selected) != n_rows:
            raise ValueError(
                "precision_selected length does not match distance length."
            )
        precision_mask = precision_selected.astype(bool)

    valid = (
        precision_mask
        & np.isfinite(distance)
        & np.isfinite(ler)
        & (ler >= 0)
    )
    rows = np.flatnonzero(valid)

    if len(rows) == 0:
        # Fall back to the currently saved best code.
        return saved_best_row(group, n_rows)

    return min(
        rows.tolist(),
        key=lambda row: (
            -int(distance[row]),
            float(ler[row]),
            float(score[row]) if np.isfinite(score[row]) else np.inf,
            int(row),
        ),
    )


def choose_row(group: h5py.Group, selection: str) -> int:
    states_ds = find_dataset(group, STATE_NAMES)
    n_rows = int(states_ds.shape[0])

    if selection == "saved":
        return saved_best_row(group, n_rows)
    if selection == "distance_then_ler":
        return distance_then_ler_row(group)

    raise ValueError(f"Unknown selection policy: {selection}")


def existing_value(
    group: h5py.Group,
    row: int,
    names: Iterable[str],
) -> float:
    array = read_1d(group, names, required=False, dtype=float)
    if array is None or row >= len(array):
        return np.nan
    value = float(array[row])
    return value if math.isfinite(value) else np.nan


def ensure_row_dataset(
    group: h5py.Group,
    name: str,
    n_rows: int,
    dtype,
    fill_value,
) -> h5py.Dataset:
    if name in group:
        dataset = group[name]
        if dataset.shape != (n_rows,):
            raise ValueError(
                f"{group.name}/{name} has shape {dataset.shape}; "
                f"expected {(n_rows,)}."
            )
        return dataset

    data = np.full(n_rows, fill_value, dtype=dtype)
    return group.create_dataset(
        name,
        data=data,
        maxshape=(None,),
        chunks=True,
    )


def update_best_selection(
    group: h5py.Group,
    row: int,
    state_edges: np.ndarray,
) -> None:
    final_best_ds = find_dataset(
        group,
        FINAL_BEST_NAMES,
        required=False,
    )
    if final_best_ds is not None:
        marker = np.asarray(final_best_ds[()])
        marker[...] = 0
        marker[row] = 1
        final_best_ds[...] = marker

    best_state = np.asarray(state_edges, dtype=np.uint32)[np.newaxis, :]
    if "best_state" in group:
        if group["best_state"].shape == best_state.shape:
            group["best_state"][:] = best_state
        else:
            del group["best_state"]
            group.create_dataset(
                "best_state",
                data=best_state,
                dtype=np.uint32,
            )
    else:
        group.create_dataset(
            "best_state",
            data=best_state,
            dtype=np.uint32,
        )

    distance = existing_value(group, row, DISTANCE_NAMES)
    ler = existing_value(group, row, LER_NAMES)
    ler_std = existing_value(group, row, LER_STD_NAMES)
    score = existing_value(group, row, SCORE_NAMES)

    group.attrs["best_row_idx"] = int(row)
    if math.isfinite(distance):
        group.attrs["best_dist"] = int(distance)
    if math.isfinite(ler):
        group.attrs["min_cost"] = float(ler)
        group.attrs["best_ler"] = float(ler)
    if math.isfinite(ler_std):
        group.attrs["best_ler_std"] = float(ler_std)
    if math.isfinite(score):
        group.attrs["best_score"] = float(score)
    group.attrs["selection_priority"] = "distance_then_ler_then_score"


def mark_running(
    hdf5_path: Path,
    run_group_path: str,
    prefix: str,
    row: int,
    budget: int,
    timestamp: str,
    selection: str,
) -> None:
    with h5py.File(hdf5_path, "r+") as h5:
        group = h5[run_group_path]
        group.attrs[f"{prefix}_status"] = "running"
        group.attrs[f"{prefix}_row_idx"] = int(row)
        group.attrs[f"{prefix}_budget"] = int(budget)
        group.attrs[f"{prefix}_started_utc"] = timestamp
        group.attrs[f"{prefix}_selection_policy"] = selection
        group.attrs[f"{prefix}_host"] = socket.gethostname()
        group.attrs[f"{prefix}_pid"] = os.getpid()
        h5.flush()


def save_result(
    hdf5_path: Path,
    run_group_path: str,
    prefix: str,
    budget: int,
    timestamp: str,
    row: int,
    state_edges: np.ndarray,
    result: dict,
    selection: str,
    update_best: bool,
) -> None:
    with h5py.File(hdf5_path, "r+") as h5:
        group = h5[run_group_path]
        states_ds = find_dataset(group, STATE_NAMES)
        n_rows = int(states_ds.shape[0])

        selected_ds = ensure_row_dataset(
            group,
            f"{prefix}_selected",
            n_rows,
            np.bool_,
            False,
        )
        ler_ds = ensure_row_dataset(
            group,
            f"{prefix}_ler",
            n_rows,
            np.float64,
            np.nan,
        )
        std_ds = ensure_row_dataset(
            group,
            f"{prefix}_ler_std",
            n_rows,
            np.float64,
            np.nan,
        )
        runtime_ds = ensure_row_dataset(
            group,
            f"{prefix}_runtime_seconds",
            n_rows,
            np.float64,
            np.nan,
        )
        completed_ds = ensure_row_dataset(
            group,
            f"{prefix}_completed_runs",
            n_rows,
            np.int64,
            -1,
        )
        failures_ds = ensure_row_dataset(
            group,
            f"{prefix}_failures",
            n_rows,
            np.int64,
            -1,
        )

        # This prefix represents the latest result for this budget.
        selected_ds[:] = False
        selected_ds[row] = True
        ler_ds[row] = float(result["ler"])
        std_ds[row] = float(result["stderr"])
        runtime_ds[row] = float(result["runtime"])
        completed_ds[row] = int(result["completed_runs"])
        failures_ds[row] = int(result["failures"])

        group.attrs[f"{prefix}_status"] = "complete"
        group.attrs[f"{prefix}_row_idx"] = int(row)
        group.attrs[f"{prefix}_budget"] = int(budget)
        group.attrs[f"{prefix}_ler"] = float(result["ler"])
        group.attrs[f"{prefix}_ler_std"] = float(result["stderr"])
        group.attrs[f"{prefix}_runtime_seconds"] = float(result["runtime"])
        group.attrs[f"{prefix}_completed_runs"] = int(
            result["completed_runs"]
        )
        group.attrs[f"{prefix}_failures"] = int(result["failures"])
        group.attrs[f"{prefix}_early_stopped"] = bool(
            result["early_stopped"]
        )
        group.attrs[f"{prefix}_timestamp_utc"] = timestamp
        group.attrs[f"{prefix}_selection_policy"] = selection
        group.attrs[f"{prefix}_state_sha256"] = state_hash(state_edges)
        group.attrs[f"{prefix}_host"] = socket.gethostname()

        history_root = group.require_group("reevaluations")
        budget_group = history_root.require_group(f"budget_{budget}")
        history_name = timestamp
        suffix = 1
        while history_name in budget_group:
            history_name = f"{timestamp}_{suffix}"
            suffix += 1
        history = budget_group.create_group(history_name)
        history.attrs["row_idx"] = int(row)
        history.attrs["budget"] = int(budget)
        history.attrs["ler"] = float(result["ler"])
        history.attrs["ler_std"] = float(result["stderr"])
        history.attrs["runtime_seconds"] = float(result["runtime"])
        history.attrs["completed_runs"] = int(result["completed_runs"])
        history.attrs["failures"] = int(result["failures"])
        history.attrs["early_stopped"] = bool(result["early_stopped"])
        history.attrs["selection_policy"] = selection
        history.attrs["state_sha256"] = state_hash(state_edges)
        history.attrs["host"] = socket.gethostname()
        history.create_dataset(
            "state",
            data=np.asarray(state_edges, dtype=np.uint32)[np.newaxis, :],
            dtype=np.uint32,
        )

        if update_best:
            update_best_selection(group, row, state_edges)

        h5.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate one repeated-run best qLDPC code."
    )
    parser.add_argument(
        "--results-root",
        default=(
            "optimization/results/"
            "logical_guided_absolute_score_gamma01_slack1_repeated_runs"
        ),
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("beam", "greedy"),
    )
    parser.add_argument("-C", required=True, type=int, choices=range(4))
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--budget", type=int, default=5_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--selection",
        choices=("saved", "distance_then_ler"),
        default="distance_then_ler",
        help=(
            "saved: use the currently saved best row; "
            "distance_then_ler: recompute the best row among existing "
            "precision-evaluated candidates."
        ),
    )
    parser.add_argument(
        "--update-best-selection",
        action="store_true",
        help=(
            "Also repair best_state/final_best/best_row_idx to the selected "
            "distance-then-LER row. The new 5e6 result is still stored separately."
        ),
    )
    parser.add_argument(
        "--run-group",
        default=None,
        help="Exact HDF5 run-group path when auto-detection is ambiguous.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if this budget already has a completed result.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.run < 1:
        raise ValueError("--run must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.budget < 1:
        raise ValueError("--budget must be positive.")

    filename = (
        f"logical_guided_{args.method}_absolute_score_gamma01_slack1_"
        f"C{args.C}_run{args.run}.hdf5"
    )
    hdf5_path = (
        Path(args.results_root)
        / args.method
        / filename
    )

    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    prefix = f"reeval_{args.budget}"
    timestamp = now_utc()

    with h5py.File(hdf5_path, "r") as h5:
        run_group_path = choose_run_group(h5, args.run_group)
        group = h5[run_group_path]

        status = group.attrs.get(f"{prefix}_status", "")
        if isinstance(status, bytes):
            status = status.decode("utf-8", errors="replace")

        if status == "complete" and not args.overwrite:
            row = group.attrs.get(f"{prefix}_row_idx", "unknown")
            ler = group.attrs.get(f"{prefix}_ler", np.nan)
            print(
                f"SKIP: {hdf5_path}\n"
                f"  {prefix} is already complete\n"
                f"  row={row}, LER={ler}\n"
                "  Use --overwrite to run it again."
            )
            return 0

        row = choose_row(group, args.selection)
        states_ds = find_dataset(group, STATE_NAMES)
        state_edges = np.asarray(states_ds[row], dtype=np.uint32)

        distance = existing_value(group, row, DISTANCE_NAMES)
        old_ler = existing_value(group, row, LER_NAMES)
        old_std = existing_value(group, row, LER_STD_NAMES)
        score = existing_value(group, row, SCORE_NAMES)

        p = float(group.attrs.get("p", noise_levels[args.C]))

    print("=" * 72)
    print("5e6 BEST-CODE RE-EVALUATION")
    print(f"File: {hdf5_path}")
    print(f"Run group: {run_group_path}")
    print(f"Method: {args.method}")
    print(f"C: {args.C} ({codes[args.C]})")
    print(f"Run: {args.run}")
    print(f"Selection policy: {args.selection}")
    print(f"Selected HDF5 row: {row}")
    print(f"Distance: {distance}")
    print(f"Stored precision LER: {old_ler} ± {old_std}")
    print(f"Stored score: {score}")
    print(f"Physical error rate p: {p}")
    print(f"Budget: {args.budget}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"State SHA256: {state_hash(state_edges)}")
    print("=" * 72, flush=True)

    mark_running(
        hdf5_path=hdf5_path,
        run_group_path=run_group_path,
        prefix=prefix,
        row=row,
        budget=args.budget,
        timestamp=timestamp,
        selection=args.selection,
    )

    state = from_edgelist(state_edges)
    _, Hx, Hz = get_code_parameters_and_matrices(state)

    wall_start = time.time()
    result = evaluate_mc(
        Hx,
        Hz,
        p,
        args.budget,
        run_label=(
            f"reeval_{args.budget}_{args.method}_"
            f"C{args.C}_run{args.run}_row{row}"
        ),
        workers=args.workers,
        batch_size=args.batch_size,
    )
    wall_runtime = time.time() - wall_start

    save_result(
        hdf5_path=hdf5_path,
        run_group_path=run_group_path,
        prefix=prefix,
        budget=args.budget,
        timestamp=timestamp,
        row=row,
        state_edges=state_edges,
        result=result,
        selection=args.selection,
        update_best=args.update_best_selection,
    )

    print("\n>>> RE-EVALUATION RESULT")
    print(f"Selected row: {row}")
    print(f"Distance: {distance}")
    print(
        f"LER: {float(result['ler']):.8f} "
        f"± {float(result['stderr']):.8f}"
    )
    print(f"Failures: {int(result['failures'])}")
    print(f"Completed trials: {int(result['completed_runs'])}")
    print(f"Evaluator runtime: {float(result['runtime']):.2f}s")
    print(f"Wall runtime: {wall_runtime:.2f}s")
    print(f"Saved prefix: {prefix}")
    print(f"Saved to: {hdf5_path}")
    print(f"Run group: {run_group_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
