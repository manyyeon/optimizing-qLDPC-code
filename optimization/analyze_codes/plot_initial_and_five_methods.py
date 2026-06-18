#!/usr/bin/env python3
"""Combine initial-code, previous-work, and current-method results in one plot.

The final plot compares:
    1. Initial code
    2. Random walk
    3. Simulated annealing
    4. Distance preprocessing + LER beam
    5. Absolute-score greedy
    6. Absolute-score beam

Selection for each absolute-score method/code family:
    1. larger final distance
    2. smaller evaluated LER
    3. smaller LER standard error
    4. smaller final score
    5. smaller run ID

If run_summary.csv contains reeval_5000000_ler and
reeval_5000000_ler_std, those columns are preferred. Otherwise the script
uses final_ler and final_ler_std.

For initial-code results, if more than one row exists for a code family,
the row with the largest completed Monte Carlo budget is used. Ties are
resolved in favor of the later CSV row, without selecting on the measured
LER.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ORDER = ["[[625,25]]", "[[1225,49]]", "[[1600,64]]", "[[2025,81]]"]

METHOD_ORDER = [
    "Initial code",
    "Random walk",
    "Simulated annealing",
    "Distance preproc. + LER | beam",
    "Absolute score | greedy",
    "Absolute score | beam",
]

METHOD_MARKERS = {
    "Initial code": "X",
    "Random walk": "o",
    "Simulated annealing": "^",
    "Distance preproc. + LER | beam": "D",
    "Absolute score | greedy": "s",
    "Absolute score | beam": "*",
}

# Small visual offsets only. The true distances remain the integer values
# stored in the CSV files.
METHOD_OFFSETS = {
    "Initial code": -0.20,
    "Random walk": -0.12,
    "Simulated annealing": -0.04,
    "Distance preproc. + LER | beam": 0.04,
    "Absolute score | greedy": 0.12,
    "Absolute score | beam": 0.20,
}

DISPLAY_LABELS = {
    "Initial code": "Initial code",
    "Random walk": "Random walk",
    "Simulated annealing": "Simulated annealing",
    "Distance preproc. + LER | beam": "Distance preproc. + LER\nbeam",
    "Absolute score | greedy": "Absolute score\ngreedy",
    "Absolute score | beam": "Absolute score\nbeam",
}


def code_to_double_brackets(code_family: str) -> str:
    text = str(code_family).strip()
    if text.startswith("[[") and text.endswith("]]"):
        return text
    if text.startswith("[") and text.endswith("]"):
        return f"[{text}]"
    return text


def seconds_to_runtime(seconds: float) -> str:
    if not np.isfinite(seconds):
        return ""

    seconds = int(round(float(seconds)))
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


def choose_evaluation_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    if "reeval_5000000_ler" in df.columns:
        reevaluated = pd.to_numeric(
            df["reeval_5000000_ler"], errors="coerce"
        )
        if reevaluated.notna().any():
            std_column = (
                "reeval_5000000_ler_std"
                if "reeval_5000000_ler_std" in df.columns
                else "final_ler_std"
            )
            return (
                "reeval_5000000_ler",
                std_column,
                "5e6 reevaluation",
            )

    return "final_ler", "final_ler_std", "stored precision evaluation"


def select_absolute_results(
    run_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    df = run_summary.copy()
    ler_column, std_column, source_label = choose_evaluation_columns(df)

    numeric_columns = [
        "final_distance",
        ler_column,
        std_column,
        "final_score",
        "runtime_seconds",
        "run",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["method", "C", "final_distance", ler_column]
    ).copy()

    selected = (
        df.sort_values(
            [
                "method",
                "C",
                "final_distance",
                ler_column,
                std_column,
                "final_score",
                "run",
            ],
            ascending=[True, True, False, True, True, True, True],
            na_position="last",
        )
        .groupby(["method", "C"], as_index=False)
        .head(1)
        .copy()
    )

    rows: list[dict] = []
    for row in selected.itertuples(index=False):
        if row.method == "greedy":
            method_label = "Absolute score | greedy"
        elif row.method == "beam":
            method_label = "Absolute score | beam"
        else:
            continue

        ler_value = getattr(row, ler_column)
        std_value = getattr(row, std_column)

        rows.append(
            {
                "code": code_to_double_brackets(row.code_family),
                "method": method_label,
                "group": "This work",
                "distance": int(row.final_distance),
                "ler": float(ler_value),
                "ler_std": float(std_value),
                "runtime": seconds_to_runtime(float(row.runtime_seconds)),
                "selected_run": int(row.run),
                "source_file": row.filename,
                "evaluation_source": source_label,
            }
        )

    return pd.DataFrame(rows), source_label


def select_initial_results(initial_results: pd.DataFrame) -> pd.DataFrame:
    """Convert initial-code evaluation rows to the plot-table schema."""
    df = initial_results.copy()
    df["_row_order"] = np.arange(len(df))

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    required = ["code_family", "d_quantum", "ler", "stderr"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Initial-results CSV is missing required columns: "
            + ", ".join(missing)
        )

    numeric_columns = [
        "d_quantum",
        "ler",
        "stderr",
        "budget",
        "completed_runs",
        "eval_runtime_seconds",
        "_row_order",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["code"] = df["code_family"].map(code_to_double_brackets)
    df = df.dropna(subset=["code", "d_quantum", "ler", "stderr"]).copy()

    # Do not select an initial-code result because its measured LER is smaller.
    # Prefer the evaluation with the largest number of completed trials.
    if "completed_runs" not in df.columns:
        df["completed_runs"] = (
            df["budget"] if "budget" in df.columns else 0
        )
    if "budget" not in df.columns:
        df["budget"] = df["completed_runs"]

    selected = (
        df.sort_values(
            ["code", "completed_runs", "budget", "_row_order"],
            ascending=[True, False, False, False],
            na_position="last",
        )
        .groupby("code", as_index=False)
        .head(1)
        .copy()
    )

    rows: list[dict] = []
    for row in selected.itertuples(index=False):
        runtime_seconds = getattr(row, "eval_runtime_seconds", np.nan)

        rows.append(
            {
                "code": row.code,
                "method": "Initial code",
                "group": "Initial",
                "distance": int(row.d_quantum),
                "ler": float(row.ler),
                "ler_std": float(row.stderr),
                "runtime": seconds_to_runtime(float(runtime_seconds)),
                "selected_run": np.nan,
                "source_file": getattr(row, "initial_file", ""),
                "evaluation_source": (
                    f"initial-code evaluation, "
                    f"{int(row.completed_runs):,} completed trials"
                ),
            }
        )

    return pd.DataFrame(rows)


def normalize_method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "code" not in result.columns:
        if "code_family" not in result.columns:
            raise ValueError(
                "Method-comparison CSV must contain 'code' or 'code_family'."
            )
        result["code"] = result["code_family"]

    result["code"] = result["code"].map(code_to_double_brackets)

    for column in ["distance", "ler", "ler_std"]:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            )

    return result


def method_sort_key(method: str) -> int:
    return (
        METHOD_ORDER.index(method)
        if method in METHOD_ORDER
        else len(METHOD_ORDER)
    )


def marker_size(marker: str) -> float:
    if marker == "*":
        return 13
    if marker == "X":
        return 9
    return 7.5


def plot_distance_vs_ler(
    df: pd.DataFrame,
    methods: list[str],
    output_prefix: Path,
) -> None:
    sub = df[
        df["method"].isin(methods) & df["code"].isin(CODE_ORDER)
    ].copy()

    sub = sub.dropna(
        subset=["code", "method", "distance", "ler", "ler_std"]
    ).copy()

    if sub.empty:
        raise ValueError("No matching rows were found for the requested plot.")

    sub["method_order"] = sub["method"].map(method_sort_key)
    sub["code"] = pd.Categorical(
        sub["code"],
        categories=CODE_ORDER,
        ordered=True,
    )
    sub = sub.sort_values(["code", "method_order"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)

    # Obtain one consistent Matplotlib color per code family.
    code_to_color: dict[str, str] = {}
    for code in CODE_ORDER:
        if not sub[sub["code"] == code].empty:
            dummy = ax.plot(
                [],
                [],
                marker="o",
                linestyle="none",
                label=code,
            )[0]
            code_to_color[code] = dummy.get_color()

    # Tiny additional offset prevents exact overlap between code-family points
    # that share both method and distance.
    code_offsets = {
        code: offset
        for code, offset in zip(
            CODE_ORDER,
            np.linspace(-0.018, 0.018, len(CODE_ORDER)),
        )
    }

    for row in sub.itertuples(index=False):
        code = str(row.code)
        method = row.method
        marker = METHOD_MARKERS.get(method, "o")

        x_value = (
            float(row.distance)
            + METHOD_OFFSETS.get(method, 0.0)
            + code_offsets.get(code, 0.0)
        )

        is_previous = row.group == "Previous work"
        is_initial = method == "Initial code"

        ax.errorbar(
            x_value,
            row.ler,
            yerr=row.ler_std,
            fmt=marker,
            color=code_to_color[code],
            markerfacecolor=(
                "none" if is_previous else code_to_color[code]
            ),
            markeredgecolor=code_to_color[code],
            markeredgewidth=1.4,
            capsize=3,
            markersize=marker_size(marker),
            linestyle="none",
            alpha=0.62 if is_initial else 0.82,
            zorder=2 if is_initial else 3,
        )

    ax.set_yscale("log")

    integer_distances = sorted(
        {int(round(value)) for value in sub["distance"].dropna()}
    )
    ax.set_xticks(integer_distances)

    ax.set_xlabel(r"HGP distance $d_Q$")
    ax.set_ylabel("Logical error rate")
    ax.set_title(
        "Distance and logical error rate of initial and optimized HGP codes"
    )
    ax.grid(True, which="both", alpha=0.25)

    code_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=code_to_color[code],
            markerfacecolor=code_to_color[code],
            markersize=8,
            label=code,
        )
        for code in CODE_ORDER
        if code in code_to_color
    ]

    method_handles = []
    for method in methods:
        marker = METHOD_MARKERS.get(method, "o")
        is_previous = method in ["Random walk", "Simulated annealing"]

        method_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                color="0.25",
                markerfacecolor="none" if is_previous else "0.25",
                markeredgewidth=1.4,
                markersize=marker_size(marker),
                label=DISPLAY_LABELS.get(method, method),
            )
        )

    first_legend = ax.legend(
        handles=code_handles,
        title="Code family",
        frameon=False,
        loc="upper right",
    )
    ax.add_artist(first_legend)

    second_legend = ax.legend(
        handles=method_handles,
        title="Method",
        frameon=False,
        loc="lower left",
        ncol=1,
    )
    ax.add_artist(second_legend)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-summary",
        required=True,
        type=Path,
        help="Repeated-run summary for absolute-score greedy and beam.",
    )
    parser.add_argument(
        "--method-comparison",
        required=True,
        type=Path,
        help=(
            "Existing comparison CSV containing random walk, simulated "
            "annealing, and distance-preprocessing + LER results."
        ),
    )
    parser.add_argument(
        "--initial-results",
        required=True,
        type=Path,
        help="CSV produced by evaluate_initial_codes.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("figures/absolute_score"),
        type=Path,
    )
    args = parser.parse_args()

    run_summary = pd.read_csv(args.run_summary)
    method_comparison = normalize_method_comparison(
        pd.read_csv(args.method_comparison)
    )
    initial_results = pd.read_csv(args.initial_results)

    absolute_rows, evaluation_source = select_absolute_results(run_summary)
    initial_rows = select_initial_results(initial_results)

    # Remove older relative-score rows and any previously inserted initial rows
    # before constructing the unified comparison table.
    updated = method_comparison[
        ~method_comparison["method"].isin(
            [
                "Weight score | greedy",
                "Weight score | beam",
                "Absolute score | greedy",
                "Absolute score | beam",
                "Initial code",
            ]
        )
    ].copy()

    combined = pd.concat(
        [updated, absolute_rows, initial_rows],
        ignore_index=True,
        sort=False,
    )

    combined["code"] = combined["code"].map(code_to_double_brackets)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = (
        args.output_dir
        / "method_comparison_initial_and_five_methods.csv"
    )
    combined.to_csv(combined_csv, index=False)

    absolute_rows.to_csv(
        args.output_dir / "selected_absolute_score_runs.csv",
        index=False,
    )
    initial_rows.to_csv(
        args.output_dir / "selected_initial_code_results.csv",
        index=False,
    )

    plot_distance_vs_ler(
        combined,
        METHOD_ORDER,
        args.output_dir
        / "ler_vs_distance_initial_and_five_methods",
    )

    print(f"Absolute-score evaluation source: {evaluation_source}")
    print("\nSelected initial-code results:")
    print(initial_rows.to_string(index=False))
    print("\nSelected absolute-score results:")
    print(absolute_rows.to_string(index=False))
    print(f"\nCombined CSV: {combined_csv}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
