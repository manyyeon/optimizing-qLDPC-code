#!/usr/bin/env python3
"""Plot repeated-run LER results for four HGP code families in one figure.

This script reads run_summary.csv and produces one figure showing the LER
values from the 10 repeated runs for greedy and beam search across all four
code families.

Default behavior
----------------
- Uses reeval_5000000_ler if available; otherwise uses final_ler.
- Plots all runs.
- Uses color = code family, marker = method.
- Filled markers indicate runs that reach the target distance for that
  code family; open markers indicate runs below the target distance.
- Overlays the median and IQR for each method/code-family group.

Optional behavior
-----------------
--only-target-distance
    Plot only the runs that reach the target distance for that family.
    This is often the fairest way to compare LER distributions, because
    runs with smaller distance usually have larger LER.

Outputs
-------
1. repeated_run_ler_all_runs.{png,pdf,svg}   or
   repeated_run_ler_target_only.{png,pdf,svg}

2. repeated_run_ler_points.csv
3. repeated_run_ler_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ORDER = [
    "[[625,25]]",
    "[[1225,49]]",
    "[[1600,64]]",
    "[[2025,81]]",
]

METHOD_ORDER = ["greedy", "beam"]

METHOD_LABELS = {
    "greedy": "Weighted score (greedy)",
    "beam": "Weighted score (beam)",
}

METHOD_MARKERS = {
    "greedy": "s",
    "beam": "*",
}

METHOD_OFFSETS = {
    "greedy": -0.14,
    "beam": 0.14,
}


def code_to_double_brackets(code_family: str) -> str:
    text = str(code_family).strip()
    if text.startswith("[[") and text.endswith("]]"):
        return text
    if text.startswith("[") and text.endswith("]"):
        return f"[{text}]"
    return text


def choose_ler_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Prefer the 5e6 reevaluation columns when available."""
    if "reeval_5000000_ler" in df.columns:
        values = pd.to_numeric(df["reeval_5000000_ler"], errors="coerce")
        if values.notna().any():
            std_col = (
                "reeval_5000000_ler_std"
                if "reeval_5000000_ler_std" in df.columns
                else "final_ler_std"
            )
            return "reeval_5000000_ler", std_col

    return "final_ler", "final_ler_std"


def build_plot_data(
    run_summary: pd.DataFrame,
    *,
    only_target_distance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df = run_summary.copy()
    df["code"] = df["code_family"].map(code_to_double_brackets)

    ler_col, std_col = choose_ler_columns(df)
    df["ler"] = pd.to_numeric(df[ler_col], errors="coerce")
    if std_col in df.columns:
        df["ler_std"] = pd.to_numeric(df[std_col], errors="coerce")
    else:
        df["ler_std"] = np.nan

    df["final_distance"] = pd.to_numeric(df["final_distance"], errors="coerce")
    if "run" in df.columns:
        df["run"] = pd.to_numeric(df["run"], errors="coerce")
    else:
        df["run"] = np.arange(len(df))

    df = df[
        df["method"].isin(METHOD_ORDER)
        & df["code"].isin(CODE_ORDER)
    ].dropna(subset=["ler", "final_distance"]).copy()

    # One shared target distance per family.
    target_by_code = df.groupby("code")["final_distance"].max().to_dict()
    df["target_distance"] = df["code"].map(target_by_code)
    df["hit_target"] = df["final_distance"] >= df["target_distance"]

    if only_target_distance:
        df = df[df["hit_target"]].copy()

    rows = []
    for (code, method), group in df.groupby(["code", "method"]):
        lers = group["ler"].astype(float)
        rows.append(
            {
                "code": code,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_runs_plotted": int(len(group)),
                "target_distance": int(group["target_distance"].iloc[0]),
                "median_ler": float(lers.median()),
                "q1_ler": float(lers.quantile(0.25)),
                "q3_ler": float(lers.quantile(0.75)),
                "min_ler": float(lers.min()),
                "max_ler": float(lers.max()),
                "n_target_hits_plotted": int(group["hit_target"].sum()),
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["code_order"] = summary["code"].map(
            {code: i for i, code in enumerate(CODE_ORDER)}
        )
        summary["method_order"] = summary["method"].map(
            {method: i for i, method in enumerate(METHOD_ORDER)}
        )
        summary = (
            summary.sort_values(["code_order", "method_order"])
            .drop(columns=["code_order", "method_order"])
            .reset_index(drop=True)
        )

    source_label = (
        "5e6 reevaluation"
        if ler_col == "reeval_5000000_ler"
        else "stored precision evaluation"
    )
    return df.reset_index(drop=True), summary, source_label


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_prefix.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        output_prefix.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def plot_repeated_run_ler(
    plot_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    output_prefix: Path,
    only_target_distance: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)

    code_to_color = {}
    for code in CODE_ORDER:
        dummy = ax.plot([], [], marker="o", linestyle="none", label=code)[0]
        code_to_color[code] = dummy.get_color()

    jitter_template = np.array(
        [-0.045, -0.035, -0.025, -0.015, -0.005,
          0.005,  0.015,  0.025,  0.035,  0.045]
    )

    for code_index, code in enumerate(CODE_ORDER):
        for method in METHOD_ORDER:
            group = plot_df[
                (plot_df["code"] == code)
                & (plot_df["method"] == method)
            ].sort_values("run")

            if group.empty:
                continue

            x_center = code_index + METHOD_OFFSETS[method]
            x_values = x_center + jitter_template[:len(group)]

            marker = METHOD_MARKERS[method]
            color = code_to_color[code]

            for x_value, row in zip(x_values, group.itertuples(index=False)):
                facecolor = color if row.hit_target else "none"
                alpha = 0.85 if row.hit_target else 0.55

                ax.plot(
                    x_value,
                    row.ler,
                    marker=marker,
                    linestyle="none",
                    color=color,
                    markerfacecolor=facecolor,
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    markersize=6.5 if method == "greedy" else 10,
                    alpha=alpha,
                    zorder=3,
                )

            sub = summary_df[
                (summary_df["code"] == code)
                & (summary_df["method"] == method)
            ]
            if not sub.empty:
                row = sub.iloc[0]
                q1 = float(row["q1_ler"])
                median = float(row["median_ler"])
                q3 = float(row["q3_ler"])

                ax.vlines(
                    x_center,
                    q1,
                    q3,
                    colors=color,
                    linewidth=2.0,
                    zorder=4,
                )
                ax.hlines(
                    median,
                    x_center - 0.05,
                    x_center + 0.05,
                    colors=color,
                    linewidth=2.2,
                    zorder=5,
                )

    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(CODE_ORDER), dtype=float))
    ax.set_xticklabels(CODE_ORDER)
    ax.set_xlabel("HGP code family")
    ax.set_ylabel("Logical error rate")
    if only_target_distance:
        ax.set_title("Repeated-run LER for runs reaching the target distance")
    else:
        ax.set_title("Repeated-run LER across 10 runs per code family")
    ax.grid(True, which="both", axis="y", alpha=0.25)

    code_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color=code_to_color[code],
            markerfacecolor=code_to_color[code],
            markersize=7,
            label=code,
        )
        for code in CODE_ORDER
    ]

    method_handles = [
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS[method],
            linestyle="none",
            color="0.2",
            markerfacecolor="0.2",
            markersize=7 if method == "greedy" else 11,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]

    fill_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color="0.35",
            markerfacecolor="0.35",
            markersize=6,
            label="run reaches target distance",
        ),
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color="0.35",
            markerfacecolor="none",
            markersize=6,
            label="run below target distance",
        ),
        plt.Line2D(
            [0], [0],
            color="0.35",
            linewidth=2.0,
            label="median and IQR of plotted runs",
        ),
    ]

    legend1 = ax.legend(
        handles=code_handles,
        title="Code family",
        frameon=False,
        loc="upper right",
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=method_handles,
        title="Method",
        frameon=False,
        loc="upper left",
    )
    ax.add_artist(legend2)

    ax.legend(
        handles=fill_handles,
        frameon=False,
        loc="lower left",
    )

    save_figure(fig, output_prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-summary",
        required=True,
        type=Path,
        help="Repeated-run summary CSV.",
    )
    parser.add_argument(
        "--only-target-distance",
        action="store_true",
        help=(
            "Plot only runs that reach the target distance for their code "
            "family. Recommended if you want a fairer LER comparison."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/absolute_score"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_summary = pd.read_csv(args.run_summary)
    plot_df, summary_df, source_label = build_plot_data(
        run_summary,
        only_target_distance=args.only_target_distance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    points_csv = args.output_dir / "repeated_run_ler_points.csv"
    summary_csv = args.output_dir / "repeated_run_ler_summary.csv"

    plot_df.to_csv(points_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    name = (
        "repeated_run_ler_target_only"
        if args.only_target_distance
        else "repeated_run_ler_all_runs"
    )

    plot_repeated_run_ler(
        plot_df,
        summary_df,
        output_prefix=args.output_dir / name,
        only_target_distance=args.only_target_distance,
    )

    print(f"LER source used: {source_label}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nPoints CSV:  {points_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Figure written to: {args.output_dir / name}")


if __name__ == "__main__":
    main()
