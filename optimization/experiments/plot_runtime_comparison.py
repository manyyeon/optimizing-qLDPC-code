#!/usr/bin/env python3
"""Plot runtime comparison for qLDPC optimization methods.

Input CSV columns:
    code,method,group,distance,ler,ler_std,runtime

Example:
    python optimization/experiments/plot_runtime_comparison.py \
      --input-csv optimization/results/method_comparison.csv \
      --codes "[[625,25]]" "[[1225,49]]" "[[1600,64]]" "[[2025,81]]" \
      --methods "Distance preproc. + LER | beam" "Weight score | greedy" "Weight score | beam" \
      --output-prefix figures/runtime_three_methods

Speedup plot:
    python optimization/experiments/plot_runtime_comparison.py \
      --input-csv optimization/results/method_comparison.csv \
      --codes "[[625,25]]" "[[1225,49]]" "[[1600,64]]" "[[2025,81]]" \
      --methods "Distance preproc. + LER | beam" "Weight score | greedy" "Weight score | beam" \
      --plot-speedup \
      --baseline-method "Distance preproc. + LER | beam" \
      --output-prefix figures/runtime_speedup_three_methods
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CODE_ORDER = ["[[625,25]]", "[[1225,49]]", "[[1600,64]]", "[[2025,81]]"]

METHOD_ORDER = [
    "Distance preproc. + LER | beam",
    "Weight score | greedy",
    "Weight score | beam",
]

DISPLAY_LABELS = {
    "Distance preproc. + LER | beam": "Distance preproc. + LER\nbeam",
    "Weight score | greedy": "Weight score\ngreedy",
    "Weight score | beam": "Weight score\nbeam",
}


def runtime_to_minutes(runtime: str) -> float:
    """Parse runtime strings like '2d 1h', '10h 42m', '38m 31s'."""
    text = str(runtime).lower()
    total = 0.0

    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(d|h|m|s)", text):
        value = float(value)
        if unit == "d":
            total += value * 24 * 60
        elif unit == "h":
            total += value * 60
        elif unit == "m":
            total += value
        elif unit == "s":
            total += value / 60

    if total == 0.0:
        raise ValueError(f"Could not parse runtime string: {runtime!r}")

    return total


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Handles duplicated pasted CSV header.
    df = df[df["code"].astype(str) != "code"].copy()

    required = {"code", "method", "runtime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df["runtime_minutes"] = df["runtime"].map(runtime_to_minutes)
    return df.reset_index(drop=True)


def filter_rows(
    df: pd.DataFrame,
    codes: list[str],
    methods: list[str],
) -> pd.DataFrame:
    out = df[df["code"].isin(codes) & df["method"].isin(methods)].copy()

    missing_methods = [m for m in methods if m not in set(out["method"])]
    if missing_methods:
        available = sorted(df["method"].unique())
        raise ValueError(
            f"Methods not found: {missing_methods}\n"
            f"Available methods: {available}"
        )

    out["code"] = pd.Categorical(out["code"], categories=codes, ordered=True)
    out["method_order"] = out["method"].map(method_sort_key)
    out = out.sort_values(["code", "method_order"]).reset_index(drop=True)

    return out


def plot_runtime_grouped_bar(
    df: pd.DataFrame,
    codes: list[str],
    methods: list[str],
    output_prefix: Path,
):
    pivot = df.pivot(index="code", columns="method", values="runtime_minutes")
    pivot = pivot.loc[codes, methods]

    x = np.arange(len(codes))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)

    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[method].values,
            width=width,
            label=DISPLAY_LABELS.get(method, method),
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(codes)
    ax.set_ylabel("Runtime (minutes, log scale)")
    ax.set_xlabel("Code family")
    ax.set_title("Runtime comparison of optimization methods")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))

    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


def plot_speedup_grouped_bar(
    df: pd.DataFrame,
    codes: list[str],
    methods: list[str],
    baseline_method: str,
    output_prefix: Path,
):
    pivot = df.pivot(index="code", columns="method", values="runtime_minutes")
    pivot = pivot.loc[codes, methods]

    speedup = pivot[baseline_method].to_frame("baseline")
    for method in methods:
        if method == baseline_method:
            continue
        speedup[method] = pivot[baseline_method] / pivot[method]

    speedup_methods = [m for m in methods if m != baseline_method]

    x = np.arange(len(codes))
    width = 0.30

    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)

    for i, method in enumerate(speedup_methods):
        offset = (i - (len(speedup_methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            speedup[method].values,
            width=width,
            label=DISPLAY_LABELS.get(method, method),
        )

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(codes)
    ax.set_ylabel(f"Speedup over {DISPLAY_LABELS.get(baseline_method, baseline_method)}")
    ax.set_xlabel("Code family")
    ax.set_title("Runtime speedup of weight-score search")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for i, code in enumerate(codes):
        for j, method in enumerate(speedup_methods):
            offset = (j - (len(speedup_methods) - 1) / 2) * width
            value = speedup.loc[code, method]
            ax.text(
                i + offset,
                value,
                f"{value:.1f}x",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))

    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument(
        "--codes",
        nargs="+",
        default=DEFAULT_CODE_ORDER,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output-prefix",
        default="figures/runtime_comparison",
        type=Path,
    )
    parser.add_argument(
        "--plot-speedup",
        action="store_true",
    )
    parser.add_argument(
        "--baseline-method",
        default="Distance preproc. + LER | beam",
    )
    args = parser.parse_args()

    df = load_results(args.input_csv)
    df = filter_rows(df, args.codes, args.methods)

    if args.plot_speedup:
        plot_speedup_grouped_bar(
            df=df,
            codes=args.codes,
            methods=args.methods,
            baseline_method=args.baseline_method,
            output_prefix=args.output_prefix,
        )
    else:
        plot_runtime_grouped_bar(
            df=df,
            codes=args.codes,
            methods=args.methods,
            output_prefix=args.output_prefix,
        )


if __name__ == "__main__":
    main()