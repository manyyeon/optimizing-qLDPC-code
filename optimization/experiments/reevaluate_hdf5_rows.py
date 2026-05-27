#!/usr/bin/env python3
"""Re-evaluate selected HDF5 candidate rows with a larger MC budget.

This script:
1. Loads selected rows from an existing run group.
2. Re-evaluates their LER with a larger Monte Carlo budget.
3. Stores the re-evaluation results in new datasets.
4. Updates best_state and min_cost based on the new results.
"""

from __future__ import annotations

import sys
import os

import argparse
from pathlib import Path

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.experiments_settings import from_edgelist, parse_edgelist
from optimization.logical_guided_search.logical_guided_eval import (
    get_code_parameters_and_matrices,
    evaluate_mc,
)


def overwrite_dataset(group, name: str, data):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def get_run_group(h5: h5py.File, code: str, run_name: str):
    if code not in h5:
        raise KeyError(f"{code!r} not found. Available codes: {list(h5.keys())}")

    code_grp = h5[code]

    if run_name not in code_grp:
        raise KeyError(
            f"{run_name!r} not found under {code!r}. "
            f"Available runs: {list(code_grp.keys())}"
        )

    return code_grp[run_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--code", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rows", required=True, nargs="+", type=int)
    parser.add_argument("-p", "--physical-error-rate", default=0.03, type=float)
    parser.add_argument("--mc-budget", default=1_000_000, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--batch-size", default=5000, type=int)
    parser.add_argument(
        "--overwrite-main-lers",
        action="store_true",
        help="Also overwrite logical_error_rates/std/runtime for the selected rows.",
    )
    args = parser.parse_args()

    rows = list(args.rows)

    # Load states first, then close the HDF5 file during long MC evaluation.
    with h5py.File(args.file, "r") as f:
        run_grp = get_run_group(f, args.code, args.run_name)

        if "states" not in run_grp:
            raise KeyError("Dataset 'states' not found in run group.")

        states_ds = run_grp["states"]

        max_row = states_ds.shape[0] - 1
        for row in rows:
            if row < 0 or row > max_row:
                raise IndexError(f"row {row} is out of range. Valid range: 0..{max_row}")

        state_edge_lists = {row: np.asarray(states_ds[row]) for row in rows}

        old_lers = {}
        if "logical_error_rates" in run_grp:
            for row in rows:
                old_lers[row] = float(run_grp["logical_error_rates"][row])

        old_dists = {}
        if "distances_quantum" in run_grp:
            for row in rows:
                old_dists[row] = float(run_grp["distances_quantum"][row])

    results = []

    for row in rows:
        print(f"\n=== Re-evaluating row {row} with MC budget {args.mc_budget} ===")

        state = from_edgelist(state_edge_lists[row])
        params, Hx, Hz = get_code_parameters_and_matrices(state)

        dist = min(params["d_classical"], params["d_T_classical"])
        print(f"Distance: {dist}")
        if row in old_lers:
            print(f"Previous stored LER: {old_lers[row]:.8g}")

        result = evaluate_mc(
            Hx=Hx,
            Hz=Hz,
            p=args.physical_error_rate,
            budget=args.mc_budget,
            run_label=f"reeval_{args.mc_budget}_row_{row}",
            workers=args.workers,
            batch_size=args.batch_size,
        )

        print(
            f"New LER: {result['ler']:.8g} ± {result['stderr']:.8g} | "
            f"failures={result['failures']} / {result['completed_runs']} | "
            f"runtime={result['runtime']:.2f}s"
        )

        results.append(
            {
                "row": row,
                "state_edge_list": state_edge_lists[row],
                "dist": dist,
                "ler": result["ler"],
                "std": result["stderr"],
                "runtime": result["runtime"],
                "failures": result["failures"],
                "completed_runs": result["completed_runs"],
                "early_stopped": result["early_stopped"],
            }
        )

    # Pick best by new LER.
    best = min(results, key=lambda r: r["ler"])

    print("\n=== Best after re-evaluation ===")
    print(f"Best row: {best['row']}")
    print(f"Distance: {best['dist']}")
    print(f"LER: {best['ler']:.8g} ± {best['std']:.8g}")

    # Write results back into the HDF5 file.
    with h5py.File(args.file, "a") as f:
        run_grp = get_run_group(f, args.code, args.run_name)

        prefix = f"reeval_{args.mc_budget}"

        overwrite_dataset(run_grp, f"{prefix}_rows", np.array([r["row"] for r in results], dtype=np.int64))
        overwrite_dataset(run_grp, f"{prefix}_distances", np.array([r["dist"] for r in results], dtype=float))
        overwrite_dataset(run_grp, f"{prefix}_logical_error_rates", np.array([r["ler"] for r in results], dtype=float))
        overwrite_dataset(run_grp, f"{prefix}_logical_error_rates_std", np.array([r["std"] for r in results], dtype=float))
        overwrite_dataset(run_grp, f"{prefix}_runtimes", np.array([r["runtime"] for r in results], dtype=float))
        overwrite_dataset(run_grp, f"{prefix}_failures", np.array([r["failures"] for r in results], dtype=np.int64))
        overwrite_dataset(run_grp, f"{prefix}_completed_runs", np.array([r["completed_runs"] for r in results], dtype=np.int64))

        # Update best_state and summary attrs based on the 1e6 result.
        best_edges = np.asarray(best["state_edge_list"], dtype=np.uint32)

        if "best_state" in run_grp:
            del run_grp["best_state"]

        run_grp.create_dataset(
            "best_state",
            data=best_edges[np.newaxis, :],
            dtype=np.uint32,
        )

        run_grp.attrs["min_cost"] = float(best["ler"])
        run_grp.attrs["min_cost_std"] = float(best["std"])
        run_grp.attrs["best_dist"] = float(best["dist"])
        run_grp.attrs["best_row_idx"] = int(best["row"])
        run_grp.attrs["best_from_reeval_budget"] = int(args.mc_budget)

        # Optional: overwrite the main row-level LER datasets for these rows.
        if args.overwrite_main_lers:
            for r in results:
                row = r["row"]
                if "logical_error_rates" in run_grp:
                    run_grp["logical_error_rates"][row] = r["ler"]
                if "logical_error_rates_std" in run_grp:
                    run_grp["logical_error_rates_std"][row] = r["std"]
                if "decoding_runtimes" in run_grp:
                    run_grp["decoding_runtimes"][row] = r["runtime"]

        print(f"\nWrote re-evaluation datasets with prefix: {prefix}")
        print(f"Updated best_state and min_cost in: /{args.code}/{args.run_name}")


if __name__ == "__main__":
    main()