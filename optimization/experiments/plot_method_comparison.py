#!/usr/bin/env python3
"""Plot method comparison by distance and verified LER.

Input CSV format:
    code,method,group,distance,ler,ler_std,runtime

Example:
    [1600,64],Random walk,Previous work,10,0.0021126,0.0000205,44h 24m
    [1600,64],Simulated annealing,Previous work,10,0.0020254,0.0000201,88h 15m
    [1600,64],Best neighbor,This work,10,0.0015472,0.0000176,9h 3m
    [1600,64],Distance preproc. + greedy,This work,11,0.0014154,0.0000168,15m
    [1600,64],Logical-guided beam,This work,12,0.0012906,0.0000161,10h 42m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_ORDER = ["Previous work", "This work"]

DEFAULT_METHOD_ORDER = [
    "Random walk",
    "Simulated annealing",
    "Greedy",
    "Beam search",
    "Distance preproc. + greedy",
    "Distance preproc. + beam",
    "Distance preproc. + Logical-guided beam",
]

method_offsets = {
    "Random walk": -0.12,
    "Simulated annealing": 0.00,
    "Greedy": 0.04,
    "Beam search": 0.08,
    "Distance preproc. + greedy": 0.12,
    "Distance preproc. + beam": 0.16,
    "Distance preproc. + Logical-guided beam": 0.2,
}

code_offsets = {
    "[[625,25]]": -0.02,
    "[[1225,49]]": -0.005,
    "[[1600,64]]": 0.005,
    "[[2025,81]]": 0.02,
}


def _method_sort_key(method: str):
    if method in DEFAULT_METHOD_ORDER:
        return DEFAULT_METHOD_ORDER.index(method)
    return len(DEFAULT_METHOD_ORDER)


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"code", "method", "group", "distance", "ler", "ler_std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["ler"] = pd.to_numeric(df["ler"], errors="coerce")
    df["ler_std"] = pd.to_numeric(df["ler_std"], errors="coerce")

    df = df.dropna(subset=["code", "method", "group", "distance", "ler"])
    df["distance"] = df["distance"].astype(int)

    return df


def collapse_previous_best(df: pd.DataFrame) -> pd.DataFrame:
    """Replace previous-work methods by the best previous-work result per code."""
    rows = []

    for code, sub in df.groupby("code", sort=False):
        prev = sub[sub["group"] == "Previous work"]
        ours = sub[sub["group"] != "Previous work"]

        if len(prev):
            best = prev.sort_values(["ler", "distance"], ascending=[True, False]).iloc[0].copy()
            best["method"] = "Best previous work"
            rows.append(best)

        for _, row in ours.iterrows():
            rows.append(row)

    return pd.DataFrame(rows)


def plot_single_code(df: pd.DataFrame, code: str, output_prefix: Path):
    sub = df[df["code"] == code].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found for code {code}")

    sub["method_order"] = sub["method"].map(_method_sort_key)
    sub = sub.sort_values(["group", "method_order", "method"])

    methods = sub["method"].tolist()
    x = np.arange(len(sub))

    group_styles = {
        "Previous work": {"alpha": 0.55, "hatch": "//"},
        "This work": {"alpha": 0.95, "hatch": ""},
    }

    fig, (ax_d, ax_ler) = plt.subplots(
        2,
        1,
        figsize=(max(8.0, 0.75 * len(sub)), 6.2),
        sharex=True,
        constrained_layout=True,
    )

    # Panel A: distance
    for group in GROUP_ORDER:
        mask = sub["group"] == group
        if not mask.any():
            continue

        style = group_styles.get(group, {"alpha": 0.8, "hatch": ""})
        ax_d.bar(
            x[mask],
            sub.loc[mask, "distance"],
            alpha=style["alpha"],
            hatch=style["hatch"],
            label=group,
        )

    ax_d.set_ylabel(r"HGP distance $d_Q$")
    ax_d.set_title(f"{code}: method comparison")
    ax_d.grid(True, axis="y", alpha=0.25)
    ax_d.legend(frameon=False)

    # Add distance labels
    for xi, d in zip(x, sub["distance"]):
        ax_d.text(xi, d + 0.08, f"{d}", ha="center", va="bottom", fontsize=8)

    # Panel B: LER
    for group in GROUP_ORDER:
        mask = sub["group"] == group
        if not mask.any():
            continue

        style = group_styles.get(group, {"alpha": 0.8, "hatch": ""})
        ax_ler.bar(
            x[mask],
            sub.loc[mask, "ler"],
            yerr=sub.loc[mask, "ler_std"],
            capsize=3,
            alpha=style["alpha"],
            hatch=style["hatch"],
            label=group,
        )

    ax_ler.set_yscale("log")
    ax_ler.set_ylabel(r"Logical error rate")
    ax_ler.set_xlabel("Method")
    ax_ler.grid(True, which="both", axis="y", alpha=0.25)

    ax_ler.set_xticks(x)
    ax_ler.set_xticklabels(methods, rotation=28, ha="right")

    # Highlight best LER
    best_idx = sub["ler"].idxmin()
    best_pos = sub.index.get_loc(best_idx)
    best_ler = sub.loc[best_idx, "ler"]
    ax_ler.scatter([best_pos], [best_ler], s=90, marker="*", zorder=5)
    ax_ler.text(
        best_pos,
        best_ler * 0.85,
        "best",
        ha="center",
        va="top",
        fontsize=8,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)

    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")


def plot_all_codes(df: pd.DataFrame, output_prefix: Path):
    codes = list(df["code"].drop_duplicates())
    n_codes = len(codes)

    fig, axes = plt.subplots(
        n_codes,
        2,
        figsize=(12.0, max(3.0 * n_codes, 4.0)),
        constrained_layout=True,
    )

    if n_codes == 1:
        axes = np.array([axes])

    for row, code in enumerate(codes):
        sub = df[df["code"] == code].copy()
        sub["method_order"] = sub["method"].map(_method_sort_key)
        sub = sub.sort_values(["group", "method_order", "method"])

        x = np.arange(len(sub))
        labels = sub["method"].tolist()

        ax_d = axes[row, 0]
        ax_ler = axes[row, 1]

        # distance
        for group in GROUP_ORDER:
            mask = sub["group"] == group
            if not mask.any():
                continue
            ax_d.bar(x[mask], sub.loc[mask, "distance"], label=group, alpha=0.55 if group == "Previous work" else 0.95)

        ax_d.set_ylabel(r"$d_Q$")
        ax_d.set_title(f"{code}: distance")
        ax_d.grid(True, axis="y", alpha=0.25)

        # ler
        for group in GROUP_ORDER:
            mask = sub["group"] == group
            if not mask.any():
                continue
            ax_ler.errorbar(
                x[mask],
                sub.loc[mask, "ler"],
                yerr=sub.loc[mask, "ler_std"],
                fmt="o",
                capsize=3,
                label=group,
                alpha=0.65 if group == "Previous work" else 0.95,
            )

        ax_ler.set_yscale("log")
        ax_ler.set_ylabel("LER")
        ax_ler.set_title(f"{code}: LER")
        ax_ler.grid(True, which="both", axis="y", alpha=0.25)

        for ax in (ax_d, ax_ler):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right")

        if row == 0:
            ax_d.legend(frameon=False)
            ax_ler.legend(frameon=False)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)

    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")


def plot_distance_vs_ler(df: pd.DataFrame, code: str, output_prefix: Path):
    sub = df[df["code"] == code].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found for code {code!r}")

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)

    for group, marker in [("Previous work", "o"), ("This work", "s")]:
        g = sub[sub["group"] == group]
        if len(g) == 0:
            continue
        ax.errorbar(
            g["distance"],
            g["ler"],
            yerr=g["ler_std"],
            fmt=marker,
            capsize=3,
            markersize=8,
            linestyle="none",
            label=group,
            alpha=0.3,
        )

        for _, row in g.iterrows():
            ax.annotate(
                row["method"],
                (row["distance"], row["ler"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

    ax.set_yscale("log")
    ax.set_xticks(sub["distance"].unique())
    ax.set_xlabel(r"HGP distance $d_Q$")
    ax.set_ylabel(r"Logical error rate")
    ax.set_title(f"{code}: LER vs. distance")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)


def plot_distance_vs_ler_all_codes(df: pd.DataFrame, output_prefix: Path):
    codes = list(df["code"].drop_duplicates())
    n_codes = len(codes)

    fig, axes = plt.subplots(
        1,
        n_codes,
        figsize=(6.2 * n_codes, 5.0),
        sharey=False,
        constrained_layout=True,
    )

    if n_codes == 1:
        axes = [axes]

    for ax, code in zip(axes, codes):
        sub = df[df["code"] == code].copy()

        for group, marker in [("Previous work", "o"), ("This work", "s")]:
            g = sub[sub["group"] == group]
            if len(g) == 0:
                continue

            ax.errorbar(
                g["distance"],
                g["ler"],
                yerr=g["ler_std"],
                fmt=marker,
                capsize=3,
                markersize=8,
                linestyle="none",
                label=group,
                alpha=0.3,
            )

            for _, row in g.iterrows():
                ax.annotate(
                    row["method"],
                    (row["distance"], row["ler"]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                )

        ax.set_yscale("log")
        ax.set_xticks(sorted(sub["distance"].unique()))
        ax.set_xlabel(r"HGP distance $d_Q$")
        ax.set_ylabel(r"Logical error rate")
        ax.set_title(f"{code}: LER vs. distance")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].legend(frameon=False)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)

    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")


def plot_distance_vs_ler_one_axis(
    df: pd.DataFrame,
    codes_to_plot: list[str],
    output_prefix: Path,
):
    sub = df[df["code"].isin(codes_to_plot)].copy()
    if len(sub) == 0:
        available = sorted(df["code"].astype(str).unique())
        raise ValueError(
            f"No rows found for codes {codes_to_plot}. "
            f"Available code values: {available}"
        )

    method_markers = {
        "Random walk": "o",
        "Simulated annealing": "^",
        "Greedy": "s",
        "Beam search": "D",
        "Logical-guided": "v",
        "Distance preproc. + greedy": "P",
        "Distance preproc. + beam": "X",
        "Distance preproc. + Logical-guided beam": "*",
    }

    # Deterministic small offsets so different code families at the same distance
    # do not completely overlap. The x-axis still represents integer distance.
    code_offsets = {
        code: offset
        for code, offset in zip(codes_to_plot, np.linspace(-0.12, 0.12, len(codes_to_plot)))
    }

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    code_colors = {
        code: color_cycle[i % len(color_cycle)]
        for i, code in enumerate(codes_to_plot)
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)

    for _, row in sub.iterrows():
        code = row["code"]
        method = row["method"]
        group = row["group"]

        x = (
        row["distance"]
        + method_offsets.get(row["method"], 0.0)
        + code_offsets.get(row["code"], 0.0)
        )
        y = row["ler"]
        yerr = row["ler_std"]

        marker = method_markers.get(method, "o")
        color = code_colors[code]

        # Previous work = hollow marker
        # This work = filled marker
        if group == "Previous work":
            markerfacecolor = color
            markeredgewidth = 1
            alpha = 0.5
        else:
            markerfacecolor = color
            markeredgewidth = 1
            alpha = 0.5

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            color=color,
            markerfacecolor=markerfacecolor,
            markeredgewidth=markeredgewidth,
            capsize=3,
            markersize=6 if marker != "*" else 10,
            linestyle="none",
            alpha=alpha,
        )

    ax.set_yscale("log")
    ax.set_xticks(sorted(sub["distance"].unique()))
    ax.set_xlabel(r"HGP distance $d_Q$")
    ax.set_ylabel(r"Logical error rate")
    ax.set_title(r"LER vs. HGP distance for optimized HGP codes")
    ax.grid(True, which="both", alpha=0.25)

    # Legend 1: code colors
    code_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color=code_colors[code],
            markerfacecolor=code_colors[code],
            markersize=8,
            label=code,
        )
        for code in codes_to_plot
    ]

    # Legend 2: method marker shapes
    methods_in_plot = [
        m for m in method_markers
        if m in set(sub["method"])
    ]
    method_handles = [
        plt.Line2D(
            [0], [0],
            marker=method_markers[m],
            linestyle="none",
            color="0.25",
            markerfacecolor="0.25",
            markersize=8 if method_markers[m] != "*" else 11,
            label=m,
        )
        for m in methods_in_plot
    ]

    leg1 = ax.legend(
        handles=code_handles,
        title="Code family",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="upper right",
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=method_handles,
        title="Method",
        frameon=False,
        fontsize=7,
        title_fontsize=9,
        loc="lower left",
    )
    ax.add_artist(leg2)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)

    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")

def collapse_best_run_per_code_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (code, method), keep the row with the lowest LER.
    Useful when the CSV contains multiple runs of the same method.
    """
    rows = []
    for (code, method), sub in df.groupby(["code", "method"], sort=False):
        best = sub.sort_values(["ler", "distance"], ascending=[True, False]).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)

def filter_methods(df: pd.DataFrame, methods: list[str] | None) -> pd.DataFrame:
    if methods is None:
        return df
    return df[df["method"].isin(methods)].copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--code", default=None, help="Plot one code only, e.g. '[[1600,64]]'")
    parser.add_argument(
        "--codes",
        nargs="+",
        default=None,
        help='Plot multiple codes on one axis, e.g. --codes "[[625,25]]" "[[1225,49]]"',
    )
    parser.add_argument("--output-prefix", default="figures/method_comparison", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help='Only keep these methods, e.g. --methods "Random walk" "Simulated annealing" "Distance preproc. + Logical-guided beam"',
    )
    args = parser.parse_args()

    df = load_results(args.input_csv)
    df = filter_methods(df, args.methods)

    if args.codes is not None:
        plot_distance_vs_ler_one_axis(
            df,
            args.codes,
            args.output_prefix,
        )
    elif args.code is not None:
        safe_code = args.code.replace("[", "").replace("]", "").replace(",", "_")
        plot_distance_vs_ler(
            df,
            args.code,
            args.output_prefix.with_name(args.output_prefix.name + f"_{safe_code}")
        )
    else:
        plot_all_codes(df, args.output_prefix)


if __name__ == "__main__":
    main()