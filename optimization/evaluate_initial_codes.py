#!/usr/bin/env python3
"""
Evaluate the four initial HGP code families with BP+OSD and save results
both to HDF5 and CSV.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from optimization.experiments_settings import (
    codes,
    load_tanner_graph,
    noise_levels,
    parse_edgelist,
    path_to_initial_codes,
    textfiles,
)
from optimization.logical_guided_search.logical_guided_eval import (
    evaluate_mc,
    get_code_parameters_and_matrices,
)


CSV_FIELDS = [
    "status",
    "code_index",
    "code_family",
    "initial_file",
    "p",
    "budget",
    "workers",
    "batch_size",
    "n_classical",
    "k_classical",
    "d_classical",
    "n_T_classical",
    "k_T_classical",
    "d_T_classical",
    "rank_H",
    "n_quantum",
    "k_quantum",
    "d_quantum",
    "ler",
    "stderr",
    "failures",
    "completed_runs",
    "early_stopped",
    "eval_runtime_seconds",
    "wall_runtime_seconds",
    "hdf5_group",
    "error",
]


def _to_python_scalar(value):
    """Convert NumPy scalar values to ordinary Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _store_attr(group: h5py.Group, name: str, value) -> None:
    """Store a scalar HDF5 attribute safely."""
    value = _to_python_scalar(value)
    group.attrs[name] = "" if value is None else value


def _resolve_initial_file(code_index: int) -> Path:
    return Path(path_to_initial_codes) / textfiles[code_index]


def evaluate_one(
    code_index: int,
    *,
    p_override: float | None,
    budget: int,
    workers: int,
    batch_size: int,
    h5_file: h5py.File,
    run_timestamp: str,
) -> dict:
    code_family = codes[code_index]
    p = float(noise_levels[code_index] if p_override is None else p_override)
    initial_file = _resolve_initial_file(code_index)

    if not initial_file.exists():
        raise FileNotFoundError(f"Initial code file not found: {initial_file}")

    print("\n" + "=" * 80)
    print(f"Evaluating initial code {code_family}")
    print(f"File: {initial_file}")
    print(f"p={p}, budget={budget}, workers={workers}, batch_size={batch_size}")

    wall_start = time.time()

    state = load_tanner_graph(str(initial_file))
    params, Hx, Hz = get_code_parameters_and_matrices(state)

    print(
        "Parameters: "
        f"classical=[{params['n_classical']},"
        f"{params['k_classical']},"
        f"{params['d_classical']}], "
        f"transpose=[{params['n_T_classical']},"
        f"{params['k_T_classical']},"
        f"{params['d_T_classical']}], "
        f"HGP=[[{params['n_quantum']},"
        f"{params['k_quantum']},"
        f"{params['d_quantum']}]]"
    )

    result = evaluate_mc(
        Hx,
        Hz,
        p,
        budget,
        run_label=f"initial_{code_index}",
        workers=workers,
        batch_size=batch_size,
    )

    wall_runtime = time.time() - wall_start

    p_tag = f"{p:g}".replace(".", "p")
    run_name = f"initial_eval_p{p_tag}_budget{budget}_{run_timestamp}"
    code_group = h5_file.require_group(code_family)

    if run_name in code_group:
        raise RuntimeError(f"HDF5 group already exists: /{code_family}/{run_name}")

    run_group = code_group.create_group(run_name)
    edges = parse_edgelist(state).astype(np.uint32)

    run_group.create_dataset(
        "initial_state",
        data=edges[np.newaxis, :],
        dtype=np.uint32,
    )

    _store_attr(run_group, "status", "ok")
    _store_attr(run_group, "code_index", code_index)
    _store_attr(run_group, "code_family", code_family)
    _store_attr(run_group, "initial_file", str(initial_file))
    _store_attr(run_group, "p", p)
    _store_attr(run_group, "budget", budget)
    _store_attr(run_group, "workers", workers)
    _store_attr(run_group, "batch_size", batch_size)

    for key, value in params.items():
        _store_attr(run_group, key, value)

    _store_attr(run_group, "ler", result["ler"])
    _store_attr(run_group, "stderr", result["stderr"])
    _store_attr(run_group, "failures", result["failures"])
    _store_attr(run_group, "completed_runs", result["completed_runs"])
    _store_attr(run_group, "early_stopped", result["early_stopped"])
    _store_attr(run_group, "eval_runtime_seconds", result["runtime"])
    _store_attr(run_group, "wall_runtime_seconds", wall_runtime)

    h5_file.flush()

    row = {
        "status": "ok",
        "code_index": code_index,
        "code_family": code_family,
        "initial_file": str(initial_file),
        "p": p,
        "budget": budget,
        "workers": workers,
        "batch_size": batch_size,
        **{key: _to_python_scalar(value) for key, value in params.items()},
        "ler": float(result["ler"]),
        "stderr": float(result["stderr"]),
        "failures": int(result["failures"]),
        "completed_runs": int(result["completed_runs"]),
        "early_stopped": bool(result["early_stopped"]),
        "eval_runtime_seconds": float(result["runtime"]),
        "wall_runtime_seconds": float(wall_runtime),
        "hdf5_group": run_group.name,
        "error": "",
    }

    print(
        f"Result: LER={row['ler']:.8g} ± {row['stderr']:.3g}, "
        f"failures={row['failures']}/{row['completed_runs']}, "
        f"eval runtime={row['eval_runtime_seconds']:.2f}s"
    )
    print(f"Saved HDF5 group: {run_group.name}")

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the four initial HGP codes with BP+OSD."
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Code-family indices to evaluate. Default: 0 1 2 3",
    )
    parser.add_argument(
        "-p",
        "--physical-error-rate",
        type=float,
        default=None,
        help=(
            "Override the configured physical error rate for every code. "
            "By default, noise_levels[index] is used."
        ),
    )
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--output-hdf5",
        type=Path,
        default=Path("optimization/results/initial_codes_evaluation.hdf5"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("optimization/results/initial_codes_evaluation.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.budget <= 0:
        raise ValueError("--budget must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    invalid = [i for i in args.indices if i < 0 or i >= len(codes)]
    if invalid:
        raise ValueError(
            f"Invalid code indices {invalid}; valid range is 0..{len(codes) - 1}."
        )

    args.output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []

    total_start = time.time()

    with h5py.File(args.output_hdf5, "a") as h5_file:
        h5_file.attrs["description"] = (
            "BP+OSD evaluation results for the unoptimized initial HGP codes"
        )
        h5_file.attrs["last_updated"] = datetime.now().isoformat()

        for code_index in args.indices:
            try:
                row = evaluate_one(
                    code_index,
                    p_override=args.physical_error_rate,
                    budget=args.budget,
                    workers=args.workers,
                    batch_size=args.batch_size,
                    h5_file=h5_file,
                    run_timestamp=run_timestamp,
                )
            except Exception as exc:
                code_family = codes[code_index]
                initial_file = str(_resolve_initial_file(code_index))
                row = {
                    "status": "error",
                    "code_index": code_index,
                    "code_family": code_family,
                    "initial_file": initial_file,
                    "p": (
                        args.physical_error_rate
                        if args.physical_error_rate is not None
                        else noise_levels[code_index]
                    ),
                    "budget": args.budget,
                    "workers": args.workers,
                    "batch_size": args.batch_size,
                    "error": repr(exc),
                }
                print(f"\nERROR while evaluating {code_family}: {exc!r}")

            rows.append(row)

    with args.output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})

    total_runtime = time.time() - total_start

    print("\n" + "=" * 80)
    print(f"Finished {len(rows)} initial-code evaluations.")
    print(f"Total wall time: {total_runtime:.2f}s")
    print(f"HDF5: {args.output_hdf5}")
    print(f"CSV:  {args.output_csv}")

    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        raise SystemExit(f"{len(failures)} evaluation(s) failed; see the CSV.")


if __name__ == "__main__":
    main()
