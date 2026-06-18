#!/usr/bin/env python3
"""Build and plot the initial/previous/absolute-score HGP comparison.

Workflow
--------
1. Select one repeated-run result per (method, code family) using the original
   search result:
       larger final distance,
       lower stored precision LER,
       lower stored precision LER stderr,
       lower score,
       lower run number.
2. Read the completed 5e6 reevaluation for that same selected run directly
   from its HDF5 file.
3. Rebuild the selected absolute-score CSV and the combined comparison CSV.
4. Sort every output by code family, then method.
5. Generate PNG, PDF, and SVG figures.

This preserves the previously selected eight runs. The 5e6 result updates
their reported LER; it does not reselect a different run based on which other
runs happen to have been reevaluated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ORDER = [
    "[[625,25]]",
    "[[1225,49]]",
    "[[1600,64]]",
    "[[2025,81]]",
]

METHOD_ORDER = [
    "Initial code",
    "Random walk",
    "Simulated annealing",
    "Distance preproc. + LER | beam",
    "Weighted score | greedy",
    "Weighted score | beam",
]

METHOD_MARKERS = {
    "Initial code": "X",
    "Random walk": "o",
    "Simulated annealing": "^",
    "Distance preproc. + LER | beam": "D",
    "Weighted score | greedy": "s",
    "Weighted score | beam": "*",
}

METHOD_OFFSETS = {
    "Initial code": -0.20,
    "Random walk": -0.12,
    "Simulated annealing": -0.04,
    "Distance preproc. + LER | beam": 0.04,
    "Weighted score | greedy": 0.12,
    "Weighted score | beam": 0.20,
}

DISPLAY_LABELS = {
    "Initial code": "Initial code",
    "Random walk": "Random walk",
    "Simulated annealing": "Simulated annealing",
    "Distance preproc. + LER | beam": (
        "Distance preproc. + LER\nbeam"
    ),
    "Weighted score | greedy": "Weighted score\ngreedy",
    "Weighted score | beam": "Weighted score\nbeam",
}

OLD_SCORE_METHOD_LABELS = {
    "Weight score | greedy",
    "Weight score | beam",
    "Weighted score | greedy",
    "Weighted score | beam",
    "Absolute score | greedy",
    "Absolute score | beam",
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


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def find_single_run_group(h5: h5py.File) -> h5py.Group:
    paths: list[str] = []

    def visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Group):
            base = name.rsplit("/", 1)[-1]
            if base.startswith("logical_guided"):
                paths.append(name)

    h5.visititems(visitor)

    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one logical-guided run group in {h5.filename}, "
            f"but found {paths}."
        )
    return h5[paths[0]]


def read_reevaluation(
    hdf5_path: Path,
    budget: int,
) -> dict:
    prefix = f"reeval_{budget}"

    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    with h5py.File(hdf5_path, "r") as h5:
        group = find_single_run_group(h5)

        status = decode_attr(group.attrs.get(f"{prefix}_status", ""))
        if status != "complete":
            raise RuntimeError(
                f"{hdf5_path}: {prefix}_status is {status!r}, not 'complete'."
            )

        required = [
            f"{prefix}_row_idx",
            f"{prefix}_ler",
            f"{prefix}_ler_std",
            f"{prefix}_runtime_seconds",
            f"{prefix}_completed_runs",
            f"{prefix}_failures",
        ]
        missing = [name for name in required if name not in group.attrs]
        if missing:
            raise KeyError(
                f"{hdf5_path}: missing reevaluation attributes {missing}."
            )

        return {
            "reeval_status": status,
            "reeval_row_idx": int(group.attrs[f"{prefix}_row_idx"]),
            "ler": float(group.attrs[f"{prefix}_ler"]),
            "ler_std": float(group.attrs[f"{prefix}_ler_std"]),
            "reeval_runtime_seconds": float(
                group.attrs[f"{prefix}_runtime_seconds"]
            ),
            "reeval_completed_runs": int(
                group.attrs[f"{prefix}_completed_runs"]
            ),
            "reeval_failures": int(group.attrs[f"{prefix}_failures"]),
            "reeval_timestamp_utc": decode_attr(
                group.attrs.get(f"{prefix}_timestamp_utc", "")
            ),
            "reeval_selection_policy": decode_attr(
                group.attrs.get(f"{prefix}_selection_policy", "")
            ),
        }


def sort_by_code_and_method(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["code"] = result["code"].map(code_to_double_brackets)

    code_rank = {code: i for i, code in enumerate(CODE_ORDER)}
    method_rank = {method: i for i, method in enumerate(METHOD_ORDER)}

    result["_code_order"] = result["code"].map(
        code_rank).fillna(len(CODE_ORDER))
    result["_method_order"] = (
        result["method"].map(method_rank).fillna(len(METHOD_ORDER))
    )

    extra_sort = ["_code_order", "_method_order"]
    if "selected_run" in result.columns:
        extra_sort.append("selected_run")

    result = (
        result.sort_values(extra_sort, kind="stable", na_position="last")
        .drop(columns=["_code_order", "_method_order"])
        .reset_index(drop=True)
    )
    return result


def select_absolute_results(
    run_summary: pd.DataFrame,
    results_root: Path,
    reeval_budget: int,
) -> pd.DataFrame:
    """Select runs using original precision results, then load their 5e6 LER."""
    df = run_summary.copy()

    numeric_columns = [
        "C",
        "run",
        "final_distance",
        "final_ler",
        "final_ler_std",
        "final_score",
        "runtime_seconds",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            raise ValueError(f"run_summary.csv is missing {column!r}.")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[
        df["method"].isin(["greedy", "beam"])
    ].dropna(
        subset=[
            "method",
            "C",
            "run",
            "final_distance",
            "final_ler",
        ]
    ).copy()

    # Preserve the original best-run selection. Do not select using 5e6 values.
    selected = (
        df.sort_values(
            [
                "method",
                "C",
                "final_distance",
                "final_ler",
                "final_ler_std",
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

    expected_pairs = {
        (method, C)
        for method in ("greedy", "beam")
        for C in range(4)
    }
    found_pairs = {
        (str(row.method), int(row.C))
        for row in selected.itertuples(index=False)
    }
    if found_pairs != expected_pairs:
        raise RuntimeError(
            "Could not select all eight method/code combinations. "
            f"Missing: {sorted(expected_pairs - found_pairs)}"
        )

    rows: list[dict] = []
    for row in selected.itertuples(index=False):
        method = str(row.method)
        method_label = (
            "Weighted score | greedy"
            if method == "greedy"
            else "Weighted score | beam"
        )

        filename = str(row.filename)
        hdf5_path = results_root / method / filename
        reeval = read_reevaluation(hdf5_path, reeval_budget)

        best_row_idx = getattr(row, "best_index", np.nan)
        if np.isfinite(best_row_idx):
            best_row_idx = int(best_row_idx)

        rows.append(
            {
                "code": code_to_double_brackets(row.code_family),
                "method": method_label,
                "group": "This work",
                "distance": int(row.final_distance),
                "ler": reeval["ler"],
                "ler_std": reeval["ler_std"],
                # Keep search runtime as the reported optimization runtime.
                "runtime": seconds_to_runtime(float(row.runtime_seconds)),
                "runtime_seconds": float(row.runtime_seconds),
                "selected_run": int(row.run),
                "selected_hdf5_row": reeval["reeval_row_idx"],
                "source_file": filename,
                "evaluation_source": (
                    f"{reeval['reeval_completed_runs']:,}-trial reevaluation"
                ),
                "reeval_status": reeval["reeval_status"],
                "reeval_runtime_seconds": reeval[
                    "reeval_runtime_seconds"
                ],
                "reeval_completed_runs": reeval[
                    "reeval_completed_runs"
                ],
                "reeval_failures": reeval["reeval_failures"],
                "reeval_timestamp_utc": reeval[
                    "reeval_timestamp_utc"
                ],
                "reeval_selection_policy": reeval[
                    "reeval_selection_policy"
                ],
            }
        )

    return sort_by_code_and_method(pd.DataFrame(rows))


def select_initial_results(initial_results: pd.DataFrame) -> pd.DataFrame:
    df = initial_results.copy()

    # Also accept an already standardized selected_initial_code_results.csv.
    standardized = {"code", "method", "group", "distance", "ler", "ler_std"}
    if standardized.issubset(df.columns):
        df = df[df["method"].astype(str) == "Initial code"].copy()
        df["code"] = df["code"].map(code_to_double_brackets)
        for column in ["distance", "ler", "ler_std"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(
            subset=["code", "distance", "ler", "ler_std"]
        ).copy()
        df = (
            df.sort_values("code", kind="stable")
            .drop_duplicates("code", keep="last")
        )
        return sort_by_code_and_method(df)

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

    return sort_by_code_and_method(pd.DataFrame(rows))


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

    sub = sort_by_code_and_method(sub)

    fig, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)

    code_to_color: dict[str, str] = {}
    for code in CODE_ORDER:
        if not sub[sub["code"] == code].empty:
            dummy = ax.plot(
                [], [], marker="o", linestyle="none", label=code
            )[0]
            code_to_color[code] = dummy.get_color()

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
            [0], [0],
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
                [0], [0],
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
        help="Repeated-run summary used to select the best run IDs.",
    )
    parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="Root containing beam/ and greedy/ HDF5 directories.",
    )
    parser.add_argument(
        "--method-comparison",
        required=True,
        type=Path,
        help=(
            "Comparison CSV containing previous work and "
            "distance-preprocessing + LER results."
        ),
    )
    parser.add_argument(
        "--initial-results",
        required=True,
        type=Path,
        help="CSV produced by evaluate_initial_codes.py.",
    )
    parser.add_argument(
        "--reeval-budget",
        type=int,
        default=5_000_000,
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

    absolute_rows = select_absolute_results(
        run_summary=run_summary,
        results_root=args.results_root,
        reeval_budget=args.reeval_budget,
    )
    initial_rows = select_initial_results(initial_results)

    # Remove every previously inserted initial/score row before rebuilding.
    updated = method_comparison[
        ~method_comparison["method"].isin(
            OLD_SCORE_METHOD_LABELS | {"Initial code"}
        )
    ].copy()

    combined = pd.concat(
        [updated, absolute_rows, initial_rows],
        ignore_index=True,
        sort=False,
    )
    combined = sort_by_code_and_method(combined)

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

    print("\nSelected 5e6 absolute-score results:")
    print(
        absolute_rows[
            [
                "code",
                "method",
                "distance",
                "ler",
                "ler_std",
                "selected_run",
                "selected_hdf5_row",
                "reeval_completed_runs",
            ]
        ].to_string(index=False)
    )
    print(f"\nCombined CSV: {combined_csv}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
