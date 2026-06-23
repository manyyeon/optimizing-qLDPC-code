#!/usr/bin/env python3
"""Extract completed 5e6-trial reevaluations from repeated-run HDF5 files.

The script scans all greedy and beam HDF5 files below --results-root,
extracts exactly one selected 5e6-trial reevaluation per file, writes a
clean CSV, and creates a paper-ready repeated-run LER figure.

Expected experiment layout
--------------------------
results_root/
    greedy/*.hdf5
    beam/*.hdf5

The filename should normally contain:
    ..._C{code_index}_run{run_number}.hdf5

The HDF5 run group should contain attributes such as:
    reeval_5000000_status
    reeval_5000000_ler
    reeval_5000000_ler_std
    reeval_5000000_row_idx
    reeval_5000000_completed_runs
    reeval_5000000_failures

Outputs
-------
reevaluated_5e6_run_summary.csv
reevaluated_5e6_ler_all_runs.{png,pdf,svg}
reevaluated_5e6_ler_target_distance.{png,pdf,svg}

Plot semantics
--------------
- Color = code family, matching the main distance-versus-LER figure.
- Marker = search method.
- Filled marker = selected code reaches the target distance for that family.
- Open marker = selected code is below the target distance.
- Short horizontal line and vertical segment = median and IQR.
- The target-distance figure includes only target-distance runs and is the
  cleaner fixed-distance comparison of LER distributions.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

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

# Initial-code reference values. These points were evaluated with 10^6
# Monte Carlo trials and are shown only as references; they are not included
# in the repeated-run median or IQR.
INITIAL_CODE_RESULTS = {
    "[[625,25]]": {
        "distance": 6,
        "ler": 1.1986e-2,
        "ler_std": 1.09e-4,
    },
    "[[1225,49]]": {
        "distance": 6,
        "ler": 1.7719e-2,
        "ler_std": 1.32e-4,
    },
    "[[1600,64]]": {
        "distance": 6,
        "ler": 1.2625e-2,
        "ler_std": 1.12e-4,
    },
    "[[2025,81]]": {
        "distance": 10,
        "ler": 2.0420e-3,
        "ler_std": 4.51e-5,
    },
}

REEVAL_PREFIX = "reeval_5000000"


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_code_family(value: str) -> str:
    text = str(value).strip()
    if text.startswith("[[") and text.endswith("]]"):
        return text
    if text.startswith("[") and text.endswith("]"):
        return f"[{text}]"
    return text


def infer_filename_metadata(path: Path) -> tuple[str | None, int | None, int | None]:
    name = path.name.lower()

    method = None
    if "greedy" in name:
        method = "greedy"
    elif "beam" in name:
        method = "beam"

    c_match = re.search(r"_c(\d+)(?:_|\.|$)", name)
    run_match = re.search(r"_run(\d+)(?:_|\.|$)", name)

    code_index = int(c_match.group(1)) if c_match else None
    run_number = int(run_match.group(1)) if run_match else None

    return method, code_index, run_number


def find_reevaluated_run_groups(h5_file: h5py.File) -> list[h5py.Group]:
    matches: list[h5py.Group] = []

    def visitor(_: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Group):
            return

        attrs = obj.attrs
        if (
            f"{REEVAL_PREFIX}_ler" in attrs
            and f"{REEVAL_PREFIX}_row_idx" in attrs
        ):
            matches.append(obj)

    h5_file.visititems(visitor)
    return matches


def value_at(group: h5py.Group, dataset_name: str, row_idx: int) -> float:
    if dataset_name not in group:
        return np.nan

    dataset = group[dataset_name]
    if row_idx < 0 or row_idx >= dataset.shape[0]:
        return np.nan

    return float(dataset[row_idx])


def bool_at(group: h5py.Group, dataset_name: str, row_idx: int) -> bool | None:
    if dataset_name not in group:
        return None

    dataset = group[dataset_name]
    if row_idx < 0 or row_idx >= dataset.shape[0]:
        return None

    return bool(dataset[row_idx])


def extract_one_file(path: Path) -> dict[str, Any]:
    method_from_name, code_index, run_number = infer_filename_metadata(path)

    with h5py.File(path, "r") as h5_file:
        groups = find_reevaluated_run_groups(h5_file)

        if not groups:
            raise ValueError("no run group with 5e6 reevaluation attributes")

        if len(groups) > 1:
            complete = [
                group
                for group in groups
                if str(
                    decode_value(
                        group.attrs.get(f"{REEVAL_PREFIX}_status", "")
                    )
                ).lower()
                == "complete"
            ]
            if len(complete) == 1:
                groups = complete
            else:
                raise ValueError(
                    f"found {len(groups)} reevaluated run groups; "
                    "could not choose one unambiguously"
                )

        group = groups[0]
        attrs = group.attrs

        status = str(
            decode_value(attrs.get(f"{REEVAL_PREFIX}_status", ""))
        )
        if status.lower() != "complete":
            raise ValueError(f"reevaluation status is {status!r}, not complete")

        row_idx = int(attrs[f"{REEVAL_PREFIX}_row_idx"])

        # The top-level parent of the run group is the code-family group.
        code_family_raw = group.name.strip("/").split("/")[0]
        code_family = normalize_code_family(code_family_raw)

        method = method_from_name
        if method is None:
            beam_width = int(attrs.get("beam_width", -1))
            method = "greedy" if beam_width == 1 else "beam"

        final_distance = value_at(group, "distances_quantum", row_idx)
        if not np.isfinite(final_distance):
            final_distance = value_at(group, "score_d_q", row_idx)
        if not np.isfinite(final_distance):
            final_distance = float(attrs.get("best_dist", np.nan))

        selected_flag = bool_at(
            group, f"{REEVAL_PREFIX}_selected", row_idx
        )
        final_best_flag = bool_at(group, "final_best", row_idx)

        row = {
            "filename": path.name,
            "filepath": str(path),
            "hdf5_group": group.name,
            "method": method,
            "C": code_index,
            "run": run_number,
            "code_family": code_family,
            "selected_row_idx": row_idx,
            "final_distance": int(final_distance)
            if np.isfinite(final_distance)
            else np.nan,
            "reeval_budget": int(
                attrs.get(f"{REEVAL_PREFIX}_budget", 5_000_000)
            ),
            "reeval_completed_runs": int(
                attrs.get(f"{REEVAL_PREFIX}_completed_runs", 0)
            ),
            "reeval_failures": int(
                attrs.get(f"{REEVAL_PREFIX}_failures", 0)
            ),
            "reeval_ler": float(attrs[f"{REEVAL_PREFIX}_ler"]),
            "reeval_ler_std": float(
                attrs.get(f"{REEVAL_PREFIX}_ler_std", np.nan)
            ),
            "reeval_runtime_seconds": float(
                attrs.get(f"{REEVAL_PREFIX}_runtime_seconds", np.nan)
            ),
            "reeval_status": status,
            "reeval_early_stopped": bool(
                attrs.get(f"{REEVAL_PREFIX}_early_stopped", False)
            ),
            "reeval_selection_policy": str(
                decode_value(
                    attrs.get(f"{REEVAL_PREFIX}_selection_policy", "")
                )
            ),
            "reeval_state_sha256": str(
                decode_value(
                    attrs.get(f"{REEVAL_PREFIX}_state_sha256", "")
                )
            ),
            "reeval_selected_flag": selected_flag,
            "final_best_flag": final_best_flag,
            "search_runtime_seconds": float(
                attrs.get("total_runtime", np.nan)
            ),
            "stored_best_distance": float(
                attrs.get("best_dist", np.nan)
            ),
            "stored_best_ler": float(
                attrs.get("best_ler", np.nan)
            ),
            "stored_best_ler_std": float(
                attrs.get("best_ler_std", np.nan)
            ),
        }

        return row


def extract_all(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(results_root.rglob("*.hdf5"))
    if not files:
        raise FileNotFoundError(
            f"No HDF5 files found below {results_root}"
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in files:
        try:
            rows.append(extract_one_file(path))
        except Exception as exc:
            errors.append(
                {
                    "filename": path.name,
                    "filepath": str(path),
                    "error": repr(exc),
                }
            )

    result = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors)

    if result.empty:
        raise RuntimeError("No completed 5e6 reevaluations were extracted.")

    result["code_family"] = result["code_family"].map(
        normalize_code_family
    )

    result["code_order"] = result["code_family"].map(
        {code: idx for idx, code in enumerate(CODE_ORDER)}
    )
    result["method_order"] = result["method"].map(
        {method: idx for idx, method in enumerate(METHOD_ORDER)}
    )

    result = (
        result.sort_values(
            ["code_order", "method_order", "run", "filename"],
            na_position="last",
        )
        .drop(columns=["code_order", "method_order"])
        .reset_index(drop=True)
    )

    return result, error_df


def validate_extraction(df: pd.DataFrame, expected_runs: int) -> None:
    print("\nExtracted run counts:")
    counts = (
        df.groupby(["code_family", "method"])
        .size()
        .rename("count")
        .reset_index()
    )
    print(counts.to_string(index=False))

    duplicates = df.duplicated(
        subset=["code_family", "method", "run"], keep=False
    )
    if duplicates.any():
        print("\nWARNING: duplicate code/method/run identifiers:")
        print(
            df.loc[
                duplicates,
                ["code_family", "method", "run", "filename"],
            ].to_string(index=False)
        )

    bad_budget = df["reeval_completed_runs"] != 5_000_000
    if bad_budget.any():
        print("\nWARNING: rows not completed to exactly 5,000,000 trials:")
        print(
            df.loc[
                bad_budget,
                [
                    "code_family",
                    "method",
                    "run",
                    "reeval_completed_runs",
                    "filename",
                ],
            ].to_string(index=False)
        )

    bad_selected = df["reeval_selected_flag"] == False  # noqa: E712
    if bad_selected.any():
        print("\nWARNING: reevaluation-selected flag is false:")
        print(
            df.loc[
                bad_selected,
                ["code_family", "method", "run", "filename"],
            ].to_string(index=False)
        )

    if len(df) != expected_runs:
        print(
            f"\nWARNING: extracted {len(df)} rows, "
            f"but --expected-runs={expected_runs}."
        )


def add_target_distance(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    target_by_code = (
        result.groupby("code_family")["final_distance"].max().to_dict()
    )
    result["target_distance"] = result["code_family"].map(
        target_by_code
    )
    result["hit_target_distance"] = (
        result["final_distance"] >= result["target_distance"]
    )

    return result


def summarize_ler(df: pd.DataFrame, target_only: bool) -> pd.DataFrame:
    data = df[df["hit_target_distance"]].copy() if target_only else df.copy()

    rows: list[dict[str, Any]] = []
    for (code, method), group in data.groupby(
        ["code_family", "method"]
    ):
        values = group["reeval_ler"].astype(float)
        rows.append(
            {
                "code_family": code,
                "method": method,
                "target_only": target_only,
                "n_runs": int(len(group)),
                "target_distance": int(group["target_distance"].iloc[0]),
                "median_ler": float(values.median()),
                "q1_ler": float(values.quantile(0.25)),
                "q3_ler": float(values.quantile(0.75)),
                "min_ler": float(values.min()),
                "max_ler": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        prefix.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        prefix.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def plot_reevaluated_ler(
    df: pd.DataFrame,
    *,
    target_only: bool,
    output_prefix: Path,
    include_initial: bool = False,
) -> None:
    points = (
        df[df["hit_target_distance"]].copy()
        if target_only
        else df.copy()
    )

    summary = summarize_ler(df, target_only=target_only)

    fig, ax = plt.subplots(
        figsize=(11.0, 6.2),
        constrained_layout=True,
    )

    # Match the code-family color semantics used in the main LER figure.
    code_to_color: dict[str, str] = {}
    for code in CODE_ORDER:
        dummy = ax.plot(
            [], [], marker="o", linestyle="none", label=code
        )[0]
        code_to_color[code] = dummy.get_color()

    jitter_template = np.linspace(-0.055, 0.055, 10)

    # Initial-code points sit at the center of each code-family category.
    # They are intentionally excluded from the repeated-run summary statistics.
    if include_initial:
        for code_idx, code in enumerate(CODE_ORDER):
            initial = INITIAL_CODE_RESULTS[code]
            color = code_to_color[code]

            ax.errorbar(
                code_idx,
                initial["ler"],
                yerr=initial["ler_std"],
                fmt="X",
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.2,
                markersize=9,
                capsize=2.5,
                elinewidth=0.9,
                linestyle="none",
                alpha=0.90,
                zorder=6,
            )

    for code_idx, code in enumerate(CODE_ORDER):
        for method in METHOD_ORDER:
            group = points[
                (points["code_family"] == code)
                & (points["method"] == method)
            ].sort_values(["run", "filename"])

            if group.empty:
                continue

            center = code_idx + METHOD_OFFSETS[method]
            jitters = jitter_template[: len(group)]
            x_values = center + jitters

            marker = METHOD_MARKERS[method]
            color = code_to_color[code]

            for x_value, row in zip(
                x_values, group.itertuples(index=False)
            ):
                filled = bool(row.hit_target_distance)
                ax.errorbar(
                    x_value,
                    row.reeval_ler,
                    yerr=row.reeval_ler_std,
                    fmt=marker,
                    color=color,
                    markerfacecolor=color if filled else "none",
                    markeredgecolor=color,
                    markeredgewidth=1.1,
                    markersize=6.5 if method == "greedy" else 9.5,
                    capsize=1.5,
                    elinewidth=0.7,
                    alpha=0.82 if filled else 0.50,
                    linestyle="none",
                    zorder=3,
                )

            summary_row = summary[
                (summary["code_family"] == code)
                & (summary["method"] == method)
            ]
            if not summary_row.empty:
                row = summary_row.iloc[0]
                q1 = float(row["q1_ler"])
                median = float(row["median_ler"])
                q3 = float(row["q3_ler"])

                ax.vlines(
                    center,
                    q1,
                    q3,
                    color=color,
                    linewidth=2.1,
                    zorder=4,
                )
                ax.hlines(
                    median,
                    center - 0.055,
                    center + 0.055,
                    color=color,
                    linewidth=2.4,
                    zorder=5,
                )

    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(CODE_ORDER), dtype=float))
    ax.set_xticklabels(CODE_ORDER)
    ax.set_xlabel("HGP code family")
    ax.set_ylabel("Logical error rate")

    if target_only:
        ax.set_title(
            r"Repeated-run LER at the maximum observed distance "
            r"($5\times10^6$ trials)"
        )
    elif include_initial:
        ax.set_title(
            r"Initial-code and repeated-run LER across 10 runs per code family "
        )
    else:
        ax.set_title(
            r"Reevaluated LER across all repeated runs "
            r"($5\times10^6$ trials)"
        )

    ax.grid(True, which="both", axis="y", alpha=0.25)

    code_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=code_to_color[code],
            markerfacecolor=code_to_color[code],
            markersize=7,
            label=code,
        )
        for code in CODE_ORDER
    ]

    method_handles = []
    if include_initial:
        method_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="X",
                linestyle="none",
                color="0.2",
                markerfacecolor="0.2",
                markersize=8,
                label=r"Initial code ($10^6$ trials)",
            )
        )

    method_handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker=METHOD_MARKERS[method],
                linestyle="none",
                color="0.2",
                markerfacecolor="0.2",
                markersize=7 if method == "greedy" else 10,
                label=METHOD_LABELS[method],
            )
            for method in METHOD_ORDER
        ]
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
        loc="upper left",
    )
    ax.add_artist(second_legend)

    if not target_only:
        status_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="0.35",
                markerfacecolor="0.35",
                markersize=6,
                label="reaches target distance",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="0.35",
                markerfacecolor="none",
                markersize=6,
                label="below target distance",
            ),
            plt.Line2D(
                [0],
                [0],
                color="0.35",
                linewidth=2.2,
                label="median and IQR",
            ),
        ]
        ax.legend(
            handles=status_handles,
            frameon=False,
            loc="lower left",
        )

    save_figure(fig, output_prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help=(
            "Root directory containing all repeated-run greedy and beam "
            "HDF5 files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=Path("figures/absolute_score"),
        type=Path,
    )
    parser.add_argument(
        "--expected-runs",
        default=80,
        type=int,
        help="Expected number of completed reevaluated run files.",
    )
    parser.add_argument(
        "--include-initial",
        action="store_true",
        help=(
            "Add the four initial-code LER reference points to the "
            "all-runs figure. Initial points are not included in the "
            "repeated-run median or IQR."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extracted, errors = extract_all(args.results_root)
    extracted = add_target_distance(extracted)

    validate_extraction(extracted, expected_runs=args.expected_runs)

    csv_path = (
        args.output_dir / "reevaluated_5e6_run_summary.csv"
    )
    error_path = (
        args.output_dir / "reevaluated_5e6_extraction_errors.csv"
    )
    all_summary_path = (
        args.output_dir / "reevaluated_5e6_ler_summary_all.csv"
    )
    target_summary_path = (
        args.output_dir
        / "reevaluated_5e6_ler_summary_target_distance.csv"
    )

    extracted.to_csv(csv_path, index=False)
    errors.to_csv(error_path, index=False)
    summarize_ler(extracted, target_only=False).to_csv(
        all_summary_path, index=False
    )
    summarize_ler(extracted, target_only=True).to_csv(
        target_summary_path, index=False
    )

    all_runs_name = (
        "reevaluated_5e6_ler_all_runs_with_initial"
        if args.include_initial
        else "reevaluated_5e6_ler_all_runs"
    )

    plot_reevaluated_ler(
        extracted,
        target_only=False,
        output_prefix=args.output_dir / all_runs_name,
        include_initial=args.include_initial,
    )
    plot_reevaluated_ler(
        extracted,
        target_only=True,
        output_prefix=(
            args.output_dir
            / "reevaluated_5e6_ler_target_distance"
        ),
        include_initial=False,
    )

    print(f"\nExtracted CSV: {csv_path}")
    print(f"Extraction errors: {error_path}")
    print(f"All-run summary: {all_summary_path}")
    print(f"Target-distance summary: {target_summary_path}")
    print(f"Figures written to: {args.output_dir}")

    if not errors.empty:
        print(
            f"\nWARNING: {len(errors)} HDF5 file(s) could not be "
            f"extracted. See {error_path}."
        )


if __name__ == "__main__":
    main()
