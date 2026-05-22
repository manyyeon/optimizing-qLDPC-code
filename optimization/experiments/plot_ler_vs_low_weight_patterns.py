#!/usr/bin/env python3
"""Plot LER versus low-weight undetectable parent-code patterns.

Each stored state is treated as one code.

For each code, this script computes:
    A_{<=d_Q+2}(H) + A_{<=d_Q+2}(H^T)

Then it plots this quantity against LER.

Example:
    python3 optimization/experiments/plot_ler_vs_low_weight_patterns.py \
      --input-file optimization/results/start_from_logical_guided.hdf5 \
      --code "[1600,64]" \
      --run-name beam_from_logical_bw2_d10_p0.03_100000screen_1000000prec_exp50_20260412_132444 \
      --max-weight 14 \
      --filter-distance 10 \
      --output-prefix figures/ler_vs_patterns_1600_64_d10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimization.analyze_codes.count_low_weight_patterns import (
    count_parent_low_weight_patterns,
)
from optimization.experiments_settings import (
    from_edgelist,
    tanner_graph_to_parity_check_matrix,
)


def _read_run(input_file: Path, code: str, run_name: str):
    with h5py.File(input_file, "r") as f:
        if code not in f:
            raise KeyError(f"{code!r} not found. Available: {list(f.keys())}")

        code_grp = f[code]

        if run_name not in code_grp:
            raise KeyError(
                f"{run_name!r} not found under {code!r}. "
                f"Available: {list(code_grp.keys())}"
            )

        grp = code_grp[run_name]

        data = {
            "states": grp["states"][:],
            "ler": grp["logical_error_rates"][:].astype(float),
            "ler_std": grp["logical_error_rates_std"][:].astype(float)
            if "logical_error_rates_std" in grp
            else np.full(len(grp["logical_error_rates"]), np.nan),
            "distance": grp["distances_quantum"][:].astype(float),
        }

        if "search_step" in grp:
            data["search_step"] = grp["search_step"][:].astype(int)
        else:
            data["search_step"] = np.arange(len(data["states"]))

        if "accepted" in grp:
            data["accepted"] = grp["accepted"][:].astype(int)
        else:
            data["accepted"] = np.ones(len(data["states"]), dtype=int)

    return data


def analyze_states(input_file: Path, code: str, run_name: str, max_weight: int, beta: float):
    data = _read_run(input_file, code, run_name)

    records = []

    for i, state_edge_list in enumerate(data["states"]):
        distance = data["distance"][i]
        ler = data["ler"][i]

        if not np.isfinite(distance) or not np.isfinite(ler) or ler <= 0:
            continue

        state = from_edgelist(state_edge_list)
        H = tanner_graph_to_parity_check_matrix(state)

        counts = count_parent_low_weight_patterns(H, max_weight=max_weight)
        counts_total = counts["counts_total"]

        d_q = int(distance)
        W_dynamic = min(d_q + 2, max_weight)

        A_leq_dynamic = int(np.sum(counts_total[1 : W_dynamic + 1]))

        A_at_d = (
            int(counts_total[d_q])
            if d_q <= max_weight
            else np.nan
        )

        weighted_score = np.nan
        if d_q <= max_weight:
            weighted_score = 0.0
            for w in range(d_q, max_weight + 1):
                weighted_score += float(counts_total[w]) * (beta ** (w - d_q))

        records.append(
            {
                "idx": i,
                "search_step": data["search_step"][i],
                "accepted": data["accepted"][i],
                "distance": d_q,
                "ler": ler,
                "ler_std": data["ler_std"][i],
                "W": W_dynamic,
                "A_leq_dplus2": A_leq_dynamic,
                "A_at_d": A_at_d,
                "weighted_score": weighted_score,
            }
        )

    return pd.DataFrame(records)


def plot_scatter(
    df: pd.DataFrame,
    output_prefix: Path,
    filter_distance: int | None,
    metric: str,
    beta: float,
):
    if filter_distance is not None:
        plot_df = df[df["distance"] == filter_distance].copy()
        title_suffix = rf"$d_Q={filter_distance}$ only"
    else:
        plot_df = df.copy()
        title_suffix = r"all distances"

    if len(plot_df) == 0:
        raise ValueError("No rows to plot after filtering.")

    if metric == "A_at_d":
        x_col = "A_at_d"
        if filter_distance is not None:
            x_label = rf"$A_{{{filter_distance}}}(H)+A_{{{filter_distance}}}(H^T)$"
        else:
            x_label = r"$A_{d_Q}(H)+A_{d_Q}(H^T)$"
        title_metric = r"$A_{d_Q}$"

    elif metric == "A_leq_dplus2":
        x_col = "A_leq_dplus2"
        if filter_distance is not None:
            x_label = rf"$A_{{\leq {filter_distance + 2}}}(H)+A_{{\leq {filter_distance + 2}}}(H^T)$"
        else:
            x_label = r"$A_{\leq d_Q+2}(H)+A_{\leq d_Q+2}(H^T)$"
        title_metric = r"$A_{\leq d_Q+2}$"

    elif metric == "weighted_score":
        x_col = "weighted_score"
        x_label = rf"Weighted low-weight score $S_{{W,\beta}}$, $\beta={beta:g}$"
        title_metric = rf"weighted score, $\beta={beta:g}$"

    else:
        raise ValueError(f"Unknown metric: {metric}")

    plot_df = plot_df[np.isfinite(plot_df[x_col]) & np.isfinite(plot_df["ler"])].copy()

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)

    sc = ax.scatter(
        plot_df[x_col],
        plot_df["ler"],
        c=plot_df["distance"],
        s=42,
        alpha=0.85,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Logical error rate")
    ax.set_yscale("log")

    if np.all(plot_df[x_col] > 0):
        ax.set_xscale("log")

    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(f"LER vs. {title_metric} ({title_suffix})")

    if filter_distance is None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"HGP distance $d_Q$")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)

    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")


def print_correlation(df: pd.DataFrame, filter_distance: int | None, metric: str):
    if filter_distance is not None:
        df = df[df["distance"] == filter_distance].copy()

    df = df[np.isfinite(df[metric]) & np.isfinite(df["ler"])].copy()

    if len(df) < 3:
        print("Not enough points to compute correlation.")
        return

    pearson = df[metric].corr(df["ler"], method="pearson")
    spearman = df[metric].corr(df["ler"], method="spearman")

    print()
    print(f"Correlation between {metric} and LER:")
    print(f"  Pearson:  {pearson:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    print()
    print("Summary:")
    print(df[["distance", "ler", "A_at_d", "A_leq_dplus2", "weighted_score"]].describe())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--code", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--max-weight",
        type=int,
        default=14,
        help="Maximum weight to count. Must be at least d_Q+2.",
    )
    parser.add_argument(
        "--filter-distance",
        type=int,
        default=None,
        help="If set, only plot codes with this HGP distance.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/ler_vs_low_weight_patterns"),
    )
    parser.add_argument(
    "--metric",
    choices=["A_at_d", "A_leq_dplus2", "weighted_score"],
    default="weighted_score",
    help="Which low-weight pattern metric to plot against LER.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Decay factor for weighted score: sum_w A_w beta^(w-d_Q).",
    )
    args = parser.parse_args()

    df = analyze_states(
        input_file=args.input_file,
        code=args.code,
        run_name=args.run_name,
        max_weight=args.max_weight,
        beta=args.beta,
    )

    csv_path = args.output_prefix.with_suffix(".csv")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    print_correlation(df, args.filter_distance, args.metric)

    plot_scatter(
        df=df,
        output_prefix=args.output_prefix,
        filter_distance=args.filter_distance,
        metric=args.metric,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()