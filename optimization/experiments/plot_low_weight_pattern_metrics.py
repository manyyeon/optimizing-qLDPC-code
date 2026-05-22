#!/usr/bin/env python3
"""Plot distance, LER, and low-weight undetectable pattern counts.

Example:
    python3 optimization/experiments/plot_low_weight_pattern_metrics.py \
      --max-weight 14 \
      --entry "Logical-guided:optimization/results/start_from_logical_guided.hdf5:[1600,64]:beam_from_logical_bw2_d10_p0.03_100000screen_1000000prec_exp50_20260412_132444" \
      --output-prefix figures/low_weight_metrics_1600_64
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimization.analyze_codes.count_low_weight_patterns import (
    count_parent_low_weight_patterns,
)
from optimization.experiments_settings import (
    from_edgelist,
    tanner_graph_to_parity_check_matrix,
)


@dataclass
class Entry:
    label: str
    path: Path
    code: str
    run_name: str | None


def parse_entry(text: str) -> Entry:
    """
    Format:
        label:path:code:run_name

    If the HDF5 group is directly /code, use run_name = "-".
    """
    parts = text.split(":", maxsplit=3)
    if len(parts) != 4:
        raise ValueError(
            "Each --entry must have format "
            "'label:path:code:run_name'. Use run_name='-' for direct code groups."
        )

    label, path, code, run_name = parts
    run_name = None if run_name == "-" else run_name

    return Entry(
        label=label,
        path=Path(path),
        code=code,
        run_name=run_name,
    )


def _get_group(f: h5py.File, entry: Entry):
    if entry.code not in f:
        raise KeyError(f"{entry.code!r} not found in {entry.path}. Available: {list(f.keys())}")

    code_grp = f[entry.code]

    if entry.run_name is None:
        return code_grp

    if entry.run_name not in code_grp:
        raise KeyError(
            f"{entry.run_name!r} not found under {entry.code!r}. "
            f"Available: {list(code_grp.keys())}"
        )

    return code_grp[entry.run_name]


def _choose_best_state_from_group(grp):
    """
    Supports two common HDF5 formats:

    1. Run group has:
        best_state

    2. Code group has:
        states, logical_error_rates, distances_quantum

    For format 2, choose the state with best LER if available.
    Otherwise choose the state with best distance.
    """
    if "best_state" in grp:
        return grp["best_state"][:], None

    if "states" not in grp:
        raise KeyError(
            "Could not find 'best_state' or 'states' dataset in this group."
        )

    states = grp["states"][:]

    if "logical_error_rates" in grp:
        ler = grp["logical_error_rates"][:].astype(float)
        valid = np.isfinite(ler) & (ler > 0)
        if np.any(valid):
            valid_indices = np.where(valid)[0]
            best_idx = valid_indices[np.argmin(ler[valid])]
            return states[best_idx], int(best_idx)

    if "distances_quantum" in grp:
        dist = grp["distances_quantum"][:].astype(float)
        valid = np.isfinite(dist)
        if np.any(valid):
            valid_indices = np.where(valid)[0]
            best_idx = valid_indices[np.argmax(dist[valid])]
            return states[best_idx], int(best_idx)

    return states[-1], len(states) - 1


def _get_ler_from_group(grp, best_idx: int | None):
    if "min_cost" in grp.attrs:
        return float(grp.attrs["min_cost"])

    if "best_ler" in grp.attrs:
        return float(grp.attrs["best_ler"])

    if "logical_error_rates" in grp:
        ler = grp["logical_error_rates"][:].astype(float)

        if best_idx is not None and 0 <= best_idx < len(ler):
            return float(ler[best_idx])

        valid = np.isfinite(ler) & (ler > 0)
        if np.any(valid):
            return float(np.min(ler[valid]))

    return np.nan


def _get_distance_from_group(grp, best_idx: int | None, counts_total):
    for key in ["best_distance", "best_dist", "distance", "d_q"]:
        if key in grp.attrs:
            return float(grp.attrs[key])

    if "distances_quantum" in grp:
        dist = grp["distances_quantum"][:].astype(float)

        if best_idx is not None and 0 <= best_idx < len(dist):
            return float(dist[best_idx])

        valid = np.isfinite(dist)
        if np.any(valid):
            return float(np.max(dist[valid]))

    # Fallback: infer from the first nonzero parent pattern count.
    for w in range(1, len(counts_total)):
        if counts_total[w] > 0:
            return float(w)

    return np.nan


def analyze_entry(entry: Entry, max_weight: int):
    with h5py.File(entry.path, "r") as f:
        grp = _get_group(f, entry)

        state_edge_list, best_idx = _choose_best_state_from_group(grp)
        state = from_edgelist(state_edge_list)
        H = tanner_graph_to_parity_check_matrix(state)

        counts = count_parent_low_weight_patterns(H, max_weight=max_weight)

        ler = _get_ler_from_group(grp, best_idx)
        distance = _get_distance_from_group(
            grp,
            best_idx,
            counts["counts_total"],
        )

    counts_total = counts["counts_total"]
    A_leq_W = int(np.sum(counts_total[1:]))

    if np.isfinite(distance):
        d_int = int(distance)
        A_at_d = int(counts_total[d_int]) if d_int < len(counts_total) else np.nan
    else:
        A_at_d = np.nan

    return {
        "label": entry.label,
        "file": str(entry.path),
        "code": entry.code,
        "run_name": entry.run_name if entry.run_name is not None else "-",
        "best_index": best_idx,
        "distance": distance,
        "ler": ler,
        "max_weight": max_weight,
        "A_leq_W": A_leq_W,
        "A_at_d": A_at_d,
    }


def _plot_positive_bar(ax, x, values, labels, ylabel, title, log=False):
    values = np.asarray(values, dtype=float)

    if log:
        plot_values = values.copy()
        zero_mask = plot_values <= 0
        plot_values[zero_mask] = 0.5
        ax.bar(x, plot_values)
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)

        for i, is_zero in enumerate(zero_mask):
            if is_zero:
                ax.text(x[i], 0.55, "0", ha="center", va="bottom", fontsize=8)
    else:
        ax.bar(x, values)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)


def plot_metrics(records, output_prefix: Path):
    df = pd.DataFrame(records)
    labels = df["label"].tolist()
    x = np.arange(len(df))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.0, 7.0),
        constrained_layout=True,
    )

    _plot_positive_bar(
        axes[0],
        x,
        df["distance"].to_numpy(),
        labels,
        ylabel=r"Distance $d_Q$",
        title="HGP distance",
        log=False,
    )

    _plot_positive_bar(
        axes[1],
        x,
        df["ler"].to_numpy(),
        labels,
        ylabel=r"Verified / estimated LER",
        title="Logical error rate",
        log=True,
    )

    _plot_positive_bar(
        axes[2],
        x,
        df["A_leq_W"].to_numpy(),
        labels,
        ylabel=r"$A_{\leq W}(H) + A_{\leq W}(H^T)$",
        title="Low-weight undetectable parent-code patterns",
        log=True,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)

    csv_path = output_prefix.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    print(df)
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help=(
            "Run entry in format 'label:path:code:run_name'. "
            "Use run_name='-' if the datasets are directly under /code."
        ),
    )
    parser.add_argument(
        "--max-weight",
        type=int,
        required=True,
        help="Maximum weight W for counting A_{<=W}.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/low_weight_pattern_metrics"),
    )
    args = parser.parse_args()

    entries = [parse_entry(e) for e in args.entry]

    records = []
    for entry in entries:
        print(f"Analyzing {entry.label}...")
        records.append(analyze_entry(entry, max_weight=args.max_weight))

    plot_metrics(records, args.output_prefix)


if __name__ == "__main__":
    main()