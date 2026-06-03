#!/usr/bin/env python3
"""Plot qLDPC method comparison as LER vs. HGP distance.

CSV columns:
    code,method,group,distance,ler,ler_std,runtime

Examples:

1) Plot all three methods:
    python plot_method_comparison.py \
      --input-csv method_comparison.csv \
      --codes "[[625,25]]" "[[1225,49]]" "[[1600,64]]" "[[2025,81]]" \
      --methods "Distance preproc. + LER | beam" "Weight score | greedy" "Weight score | beam" \
      --output-prefix figures/three_method_ler_vs_distance

2) Plot previous work plus the best method among the three for each code:
    python plot_method_comparison.py \
      --input-csv method_comparison.csv \
      --codes "[[625,25]]" "[[1225,49]]" "[[1600,64]]" "[[2025,81]]" \
      --previous-methods "Random walk" "Simulated annealing" \
      --best-this-methods "Distance preproc. + LER | beam" "Weight score | greedy" "Weight score | beam" \
      --this-label "Best from this work" \
      --output-prefix figures/ler_vs_distance_best_among_three
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,

    # Better font embedding for PDF/PS output
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    # Paper-friendly sizes
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


DEFAULT_CODE_ORDER = ["[[625,25]]", "[[1225,49]]", "[[1600,64]]", "[[2025,81]]"]

METHOD_ORDER = [
    "Random walk",
    "Simulated annealing",
    "Greedy",
    "Beam search",
    "Distance preproc. + greedy",
    "Distance preproc. + beam",
    "Distance preproc. + LER | beam",
    "Weight score | greedy",
    "Weight score | beam",
    "Best from this work",
]

METHOD_MARKERS = {
    "Random walk": "o",
    "Simulated annealing": "^",
    "Greedy": "s",
    "Beam search": "D",
    "Distance preproc. + greedy": "P",
    "Distance preproc. + beam": "X",
    "Distance preproc. + LER | beam": "D",
    "Weight score | greedy": "s",
    "Weight score | beam": "*",
    "Best from this work": "*",
}

METHOD_OFFSETS = {
    "Random walk": -0.18,
    "Simulated annealing": -0.10,
    "Greedy": -0.05,
    "Beam search": 0.00,
    "Distance preproc. + greedy": 0.05,
    "Distance preproc. + beam": 0.10,
    "Distance preproc. + LER | beam": -0.10,
    "Weight score | greedy": 0.00,
    "Weight score | beam": 0.10,
    "Best from this work": 0.10,
}

DISPLAY_LABELS = {
    "Distance preproc. + LER | beam": "Distance preproc. + LER\nbeam",
    "Weight score | greedy": "Weight score\ngreedy",
    "Weight score | beam": "Weight score\nbeam",
    "Best from this work": "Best from\nthis work",
}


def method_sort_key(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Handles duplicated CSV header pasted into the body.
    df = df[df["code"].astype(str) != "code"].copy()

    required = {"code", "method", "group", "distance", "ler", "ler_std", "runtime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["ler"] = pd.to_numeric(df["ler"], errors="coerce")
    df["ler_std"] = pd.to_numeric(df["ler_std"], errors="coerce")
    df = df.dropna(subset=["code", "method", "group", "distance", "ler", "ler_std"])
    df["distance"] = df["distance"].astype(int)

    return df.reset_index(drop=True)


def filter_methods(df: pd.DataFrame, methods: list[str] | None) -> pd.DataFrame:
    if methods is None:
        return df.copy()

    out = df[df["method"].isin(methods)].copy()
    missing = [m for m in methods if m not in set(out["method"])]
    if missing:
        available = sorted(df["method"].unique())
        raise ValueError(f"Methods not found: {missing}\nAvailable methods: {available}")

    return out


def select_previous_plus_best_this(
    df: pd.DataFrame,
    codes: list[str],
    previous_methods: list[str],
    best_this_methods: list[str],
    this_label: str,
) -> pd.DataFrame:
    """For each code, keep previous methods plus best row among candidate this-work methods.

    Best rule:
        1. larger distance is better
        2. if tied, smaller LER is better
        3. if tied, smaller LER std is better
    """
    selected: list[pd.DataFrame] = []

    for code in codes:
        sub = df[df["code"] == code].copy()

        prev = sub[sub["method"].isin(previous_methods)].copy()
        if not prev.empty:
            selected.append(prev)

        candidates = sub[
            (sub["group"] == "This work") & (sub["method"].isin(best_this_methods))
        ].copy()

        if candidates.empty:
            print(f"[warning] {code}: no candidate rows found among {best_this_methods}.")
            continue

        best = candidates.sort_values(
            ["distance", "ler", "ler_std"],
            ascending=[False, True, True],
        ).head(1).copy()

        original_method = best.iloc[0]["method"]
        best["original_method"] = original_method
        best["method"] = this_label
        selected.append(best)

        print(
            f"{code}: selected {original_method} "
            f"(d={int(best.iloc[0]['distance'])}, "
            f"LER={best.iloc[0]['ler']:.6g}, "
            f"runtime={best.iloc[0]['runtime']})"
        )

    if not selected:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(selected, ignore_index=True)


def runtime_to_minutes(runtime: str) -> float:
    """Parse strings like '2d 1h', '38m 31s', '1h 23m' into minutes."""
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
    return total


def plot_distance_vs_ler(
    df: pd.DataFrame,
    codes: list[str],
    output_prefix: Path,
    title: str,
    annotate_runtime: bool,
    annotate_selected_method: bool,
):
    sub = df[df["code"].isin(codes)].copy()
    if sub.empty:
        available = sorted(df["code"].astype(str).unique())
        raise ValueError(f"No rows found for codes={codes}. Available codes: {available}")

    sub["code"] = pd.Categorical(sub["code"], categories=codes, ordered=True)
    sub["method_order"] = sub["method"].map(method_sort_key)
    sub = sub.sort_values(["code", "method_order", "method"]).reset_index(drop=True)

    # Very small offsets keep markers at the same integer distance visible.
    code_offsets = {
        code: offset for code, offset in zip(codes, np.linspace(-0.025, 0.025, len(codes)))
    }

    fig, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)

    # Use Matplotlib's default color cycle for code families.
    code_to_color = {}
    for code in codes:
        if not sub[sub["code"] == code].empty:
            dummy = ax.plot([], [], marker="o", linestyle="none", label=code)[0]
            code_to_color[code] = dummy.get_color()

    for row in sub.itertuples(index=False):
        code = str(row.code)
        method = row.method
        marker = METHOD_MARKERS.get(method, "o")
        x = row.distance + METHOD_OFFSETS.get(method, 0.0) + code_offsets.get(code, 0.0)

        is_previous = getattr(row, "group") == "Previous work"
        ax.errorbar(
            x,
            row.ler,
            yerr=row.ler_std,
            fmt=marker,
            color=code_to_color[code],
            markerfacecolor="none" if is_previous else code_to_color[code],
            markeredgewidth=1.3,
            capsize=3,
            markersize=7 if marker != "*" else 12,
            linestyle="none",
            alpha=0.70,
        )

        label_parts = []
        if annotate_runtime:
            label_parts.append(str(row.runtime))
        if annotate_selected_method and hasattr(row, "original_method") and pd.notna(row.original_method):
            label_parts.append(str(row.original_method))

        if label_parts:
            ax.annotate(
                "\n".join(label_parts),
                (x, row.ler),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_yscale("log")
    ax.set_xticks(sorted(sub["distance"].unique()))
    ax.set_xlabel(r"HGP distance $d_Q$")
    ax.set_ylabel("Logical error rate")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    # Code-family legend.
    code_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color=code_to_color[code],
            markerfacecolor=code_to_color[code],
            markersize=8,
            label=code,
        )
        for code in codes
        if code in code_to_color
    ]

    # Method legend.
    methods_in_plot = sorted(list(dict.fromkeys(sub["method"])), key=method_sort_key)
    method_handles = [
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS.get(method, "o"),
            linestyle="none",
            color="0.25",
            markerfacecolor="none" if method in ["Random walk", "Simulated annealing"] else "0.25",
            markeredgewidth=1.3,
            markersize=8 if METHOD_MARKERS.get(method, "o") != "*" else 12,
            label=DISPLAY_LABELS.get(method, method),
        )
        for method in methods_in_plot
    ]

    leg1 = ax.legend(
        handles=code_handles,
        title="Code family",
        frameon=False,
        loc="upper right",
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=method_handles,
        title="Method",
        frameon=False,
        loc="lower left",
    )
    ax.add_artist(leg2)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")

    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {svg_path}")


def make_runtime_summary(df: pd.DataFrame, output_csv: Path) -> None:
    out = df.copy()
    out["runtime_minutes"] = out["runtime"].map(runtime_to_minutes)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument(
        "--codes",
        nargs="+",
        default=DEFAULT_CODE_ORDER,
        help='Codes to plot, e.g. --codes "[[625,25]]" "[[1225,49]]"',
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help='Methods to plot, e.g. --methods "Distance preproc. + LER | beam" "Weight score | greedy" "Weight score | beam"',
    )
    parser.add_argument(
        "--previous-methods",
        nargs="+",
        default=None,
        help='Previous-work methods to include with --best-this-methods.',
    )
    parser.add_argument(
        "--best-this-methods",
        nargs="+",
        default=None,
        help="Candidate this-work methods. For each code, plot only the best among these.",
    )
    parser.add_argument(
        "--this-label",
        default="Best from this work",
        help='Legend label for the selected best this-work point.',
    )
    parser.add_argument(
        "--title",
        default="LER vs. HGP distance for optimized HGP codes",
    )
    parser.add_argument("--output-prefix", default="figures/method_comparison", type=Path)
    parser.add_argument("--annotate-runtime", action="store_true")
    parser.add_argument(
        "--annotate-selected-method",
        action="store_true",
        help="Annotate which original method was selected for each best-this-work point.",
    )
    parser.add_argument(
        "--runtime-summary-csv",
        default=None,
        type=Path,
        help="Optional path to save the plotted rows with parsed runtime_minutes.",
    )
    args = parser.parse_args()

    df = load_results(args.input_csv)

    if args.best_this_methods is not None:
        previous_methods = args.previous_methods or ["Random walk", "Simulated annealing"]
        plot_df = select_previous_plus_best_this(
            df=df,
            codes=args.codes,
            previous_methods=previous_methods,
            best_this_methods=args.best_this_methods,
            this_label=args.this_label,
        )
    else:
        plot_df = filter_methods(df, args.methods)

    plot_distance_vs_ler(
        df=plot_df,
        codes=args.codes,
        output_prefix=args.output_prefix,
        title=args.title,
        annotate_runtime=args.annotate_runtime,
        annotate_selected_method=args.annotate_selected_method,
    )

    if args.runtime_summary_csv is not None:
        plotted = plot_df[plot_df["code"].isin(args.codes)].copy()
        make_runtime_summary(plotted, args.runtime_summary_csv)


if __name__ == "__main__":
    main()
