#!/usr/bin/env python3
"""Plot per-run runtime and repeated-run reliability for weighted-score search.

Outputs
-------
1. weighted_score_runtime_per_run.{png,pdf,svg}
   Median runtime over repeated runs, with IQR error bars.

2. weighted_score_success_rate.{png,pdf,svg}
   Fraction of runs reaching the maximum observed distance for each code family.

3. weighted_score_runtime_summary.csv
4. weighted_score_reliability_summary.csv

The optional method-comparison CSV is used only to overlay directly measured
reference runtimes:
  * Distance preprocessing + LER beam for all code families.
  * Random walk and simulated annealing only for [[625,25]].

This deliberately excludes unmeasured/estimated random-walk and
simulated-annealing runtimes for the larger code families.
"""

from __future__ import annotations

import argparse
import re
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
    "greedy": -0.10,
    "beam": 0.10,
}

BASELINE_LABELS = {
    "Distance preproc. + LER | beam": "Distance preproc. + LER beam",
    "Random walk": "Random walk",
    "Simulated annealing": "Simulated annealing",
}

BASELINE_MARKERS = {
    "Distance preproc. + LER | beam": "D",
    "Random walk": "o",
    "Simulated annealing": "^",
}

BASELINE_OFFSETS = {
    "Distance preproc. + LER | beam": 0.28,
    "Random walk": -0.32,
    "Simulated annealing": -0.22,
}

# Grayscale styling is used for the runtime and reliability figures.
# Color remains reserved for code-family identification in the main
# distance-versus-LER figure.
METHOD_STYLES = {
    "greedy": {
        "color": "0.15",
        "markerfacecolor": "0.15",
        "markeredgecolor": "0.15",
    },
    "beam": {
        "color": "0.45",
        "markerfacecolor": "0.45",
        "markeredgecolor": "0.45",
    },
}

BASELINE_STYLES = {
    "Distance preproc. + LER | beam": {
        "color": "0.10",
        "markerfacecolor": "0.10",
        "markeredgecolor": "0.10",
    },
    "Random walk": {
        "color": "0.40",
        "markerfacecolor": "none",
        "markeredgecolor": "0.40",
    },
    "Simulated annealing": {
        "color": "0.65",
        "markerfacecolor": "none",
        "markeredgecolor": "0.65",
    },
}

RELIABILITY_STYLES = {
    "greedy": {
        "facecolor": "0.25",
        "edgecolor": "0.10",
        "hatch": "",
    },
    "beam": {
        "facecolor": "0.75",
        "edgecolor": "0.20",
        "hatch": "///",
    },
}


def code_to_double_brackets(code_family: str) -> str:
    text = str(code_family).strip()
    if text.startswith("[[") and text.endswith("]]"):
        return text
    if text.startswith("[") and text.endswith("]"):
        return f"[{text}]"
    return text


def format_runtime(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--"

    total_seconds = int(round(float(seconds)))
    days, remainder = divmod(total_seconds, 86_400)
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


def parse_runtime_string(value: object) -> float:
    """Convert strings such as '2d 1h', '40h', or '17m 30s' to seconds."""
    if value is None or pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if not text or text in {"-", "--", "nan", "none"}:
        return np.nan

    units = {
        "d": 86_400,
        "h": 3_600,
        "m": 60,
        "s": 1,
    }

    total = 0.0
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*([dhms])", text)
    if not matches:
        return np.nan

    for number, unit in matches:
        total += float(number) * units[unit]
    return total


def choose_runtime_column(df: pd.DataFrame) -> pd.Series:
    """Prefer runtime_seconds and fall back to total_runtime."""
    runtime = pd.Series(np.nan, index=df.index, dtype=float)

    if "runtime_seconds" in df.columns:
        runtime = pd.to_numeric(df["runtime_seconds"], errors="coerce")

    if "total_runtime" in df.columns:
        fallback = pd.to_numeric(df["total_runtime"], errors="coerce")
        runtime = runtime.fillna(fallback)

    return runtime


def build_runtime_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    df = run_summary.copy()
    df["code"] = df["code_family"].map(code_to_double_brackets)
    df["runtime_seconds_clean"] = choose_runtime_column(df)
    df["final_distance"] = pd.to_numeric(
        df["final_distance"], errors="coerce"
    )

    df = df[
        df["method"].isin(METHOD_ORDER)
        & df["code"].isin(CODE_ORDER)
    ].copy()

    df = df.dropna(subset=["runtime_seconds_clean"])

    rows: list[dict] = []
    for (code, method), group in df.groupby(["code", "method"]):
        runtimes = group["runtime_seconds_clean"].astype(float)

        q1 = float(runtimes.quantile(0.25))
        median = float(runtimes.median())
        q3 = float(runtimes.quantile(0.75))

        rows.append(
            {
                "code": code,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_runs": int(len(runtimes)),
                "q1_runtime_seconds": q1,
                "median_runtime_seconds": median,
                "q3_runtime_seconds": q3,
                "iqr_runtime_seconds": q3 - q1,
                "mean_runtime_seconds": float(runtimes.mean()),
                "total_runtime_seconds": float(runtimes.sum()),
                "q1_runtime": format_runtime(q1),
                "median_runtime": format_runtime(median),
                "q3_runtime": format_runtime(q3),
                "total_runtime": format_runtime(float(runtimes.sum())),
            }
        )

    result = pd.DataFrame(rows)
    result["code_order"] = result["code"].map(
        {code: index for index, code in enumerate(CODE_ORDER)}
    )
    result["method_order"] = result["method"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )

    return (
        result.sort_values(["code_order", "method_order"])
        .drop(columns=["code_order", "method_order"])
        .reset_index(drop=True)
    )


def build_reliability_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Measure success against the best distance seen by either score method."""
    df = run_summary.copy()
    df["code"] = df["code_family"].map(code_to_double_brackets)
    df["final_distance"] = pd.to_numeric(
        df["final_distance"], errors="coerce"
    )

    df = df[
        df["method"].isin(METHOD_ORDER)
        & df["code"].isin(CODE_ORDER)
    ].dropna(subset=["final_distance"]).copy()

    # One common target per code family prevents a method from receiving
    # 100% success merely because it never reached the other method's maximum.
    target_by_code = df.groupby("code")["final_distance"].max().to_dict()

    rows: list[dict] = []
    for (code, method), group in df.groupby(["code", "method"]):
        target_distance = float(target_by_code[code])
        distances = group["final_distance"].astype(float)
        success = distances >= target_distance

        rows.append(
            {
                "code": code,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "target_distance": int(target_distance),
                "n_runs": int(len(group)),
                "n_success": int(success.sum()),
                "success_rate": float(success.mean()),
                "success_percent": 100.0 * float(success.mean()),
                "distance_median": float(distances.median()),
                "distance_min": float(distances.min()),
                "distance_max": float(distances.max()),
            }
        )

    result = pd.DataFrame(rows)
    result["code_order"] = result["code"].map(
        {code: index for index, code in enumerate(CODE_ORDER)}
    )
    result["method_order"] = result["method"].map(
        {method: index for index, method in enumerate(METHOD_ORDER)}
    )

    return (
        result.sort_values(["code_order", "method_order"])
        .drop(columns=["code_order", "method_order"])
        .reset_index(drop=True)
    )


def load_direct_baseline_runtimes(
    method_comparison_path: Path | None,
) -> pd.DataFrame:
    """Load only directly measured baseline runtimes used in the paper."""
    if method_comparison_path is None:
        return pd.DataFrame(
            columns=["code", "method", "runtime_seconds"]
        )

    df = pd.read_csv(method_comparison_path).copy()

    if "code" not in df.columns:
        if "code_family" not in df.columns:
            raise ValueError(
                "Method-comparison CSV must contain 'code' or 'code_family'."
            )
        df["code"] = df["code_family"]

    df["code"] = df["code"].map(code_to_double_brackets)

    if "runtime_seconds" in df.columns:
        df["runtime_seconds_clean"] = pd.to_numeric(
            df["runtime_seconds"], errors="coerce"
        )
    else:
        df["runtime_seconds_clean"] = np.nan

    if "runtime" in df.columns:
        parsed = df["runtime"].map(parse_runtime_string)
        df["runtime_seconds_clean"] = (
            df["runtime_seconds_clean"].fillna(parsed)
        )

    allowed = (
        (df["method"] == "Distance preproc. + LER | beam")
        | (
            df["method"].isin(["Random walk", "Simulated annealing"])
            & (df["code"] == "[[625,25]]")
        )
    )

    result = df.loc[
        allowed,
        ["code", "method", "runtime_seconds_clean"],
    ].rename(columns={"runtime_seconds_clean": "runtime_seconds"})

    return result.dropna(subset=["runtime_seconds"]).copy()


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


def plot_runtime_per_run(
    runtime_summary: pd.DataFrame,
    baseline_runtimes: pd.DataFrame,
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)

    x_base = np.arange(len(CODE_ORDER), dtype=float)

    for method in METHOD_ORDER:
        sub = runtime_summary[
            runtime_summary["method"] == method
        ].set_index("code")

        x_values: list[float] = []
        medians_hours: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []

        for code_index, code in enumerate(CODE_ORDER):
            if code not in sub.index:
                continue

            row = sub.loc[code]
            median = float(row["median_runtime_seconds"])
            q1 = float(row["q1_runtime_seconds"])
            q3 = float(row["q3_runtime_seconds"])

            x_values.append(
                code_index + METHOD_OFFSETS[method]
            )
            medians_hours.append(median / 3_600.0)
            lower_errors.append((median - q1) / 3_600.0)
            upper_errors.append((q3 - median) / 3_600.0)

        style = METHOD_STYLES[method]
        ax.errorbar(
            x_values,
            medians_hours,
            yerr=np.vstack([lower_errors, upper_errors]),
            marker=METHOD_MARKERS[method],
            linestyle="none",
            color=style["color"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["markeredgecolor"],
            capsize=4,
            markeredgewidth=1.3,
            markersize=8 if method == "greedy" else 12,
            label=f"{METHOD_LABELS[method]}: median and IQR",
        )

    if not baseline_runtimes.empty:
        for baseline_method in [
            "Distance preproc. + LER | beam",
            "Random walk",
            "Simulated annealing",
        ]:
            sub = baseline_runtimes[
                baseline_runtimes["method"] == baseline_method
            ]

            if sub.empty:
                continue

            x_values: list[float] = []
            runtime_hours: list[float] = []

            for row in sub.itertuples(index=False):
                if row.code not in CODE_ORDER:
                    continue
                code_index = CODE_ORDER.index(row.code)
                x_values.append(
                    code_index + BASELINE_OFFSETS[baseline_method]
                )
                runtime_hours.append(
                    float(row.runtime_seconds) / 3_600.0
                )

            style = BASELINE_STYLES[baseline_method]
            ax.plot(
                x_values,
                runtime_hours,
                marker=BASELINE_MARKERS[baseline_method],
                linestyle="none",
                color=style["color"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=1.4,
                markersize=8,
                label=(
                    f"{BASELINE_LABELS[baseline_method]}: "
                    "single measured run"
                ),
            )

    ax.set_yscale("log")
    ax.set_xticks(x_base)
    ax.set_xticklabels(CODE_ORDER)
    ax.set_xlabel("HGP code family")
    ax.set_ylabel("Search runtime per run (hours)")
    ax.set_title("Typical search cost per independent run")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, columnspacing=1.2, handletextpad=0.6)

    save_figure(fig, output_prefix)


def plot_success_rate(
    reliability_summary: pd.DataFrame,
    output_prefix: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)

    x_base = np.arange(len(CODE_ORDER), dtype=float)
    width = 0.34

    for method_index, method in enumerate(METHOD_ORDER):
        sub = reliability_summary[
            reliability_summary["method"] == method
        ].set_index("code")

        heights: list[float] = []
        labels: list[str] = []

        for code in CODE_ORDER:
            if code not in sub.index:
                heights.append(np.nan)
                labels.append("")
                continue

            row = sub.loc[code]
            heights.append(float(row["success_percent"]))
            labels.append(
                f"{int(row['n_success'])}/{int(row['n_runs'])}"
            )

        positions = x_base + (
            -width / 2 if method_index == 0 else width / 2
        )

        style = RELIABILITY_STYLES[method]
        bars = ax.bar(
            positions,
            heights,
            width=width,
            linewidth=1.0,
            hatch=style["hatch"],
            label=METHOD_LABELS[method],
        )

        for bar, label, height in zip(bars, labels, heights):
            if not np.isfinite(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2.0,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    targets = (
        reliability_summary[["code", "target_distance"]]
        .drop_duplicates()
        .set_index("code")["target_distance"]
        .to_dict()
    )
    tick_labels = [
        f"{code}\ntarget $d_Q={targets.get(code, '?')}$"
        for code in CODE_ORDER
    ]

    ax.set_xticks(x_base)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 112)
    ax.set_xlabel("HGP code family")
    ax.set_ylabel("Runs reaching target distance (%)")
    ax.set_title("Repeated-run reliability")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    save_figure(fig, output_prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-summary",
        required=True,
        type=Path,
        help="CSV containing the ten greedy and ten beam runs per family.",
    )
    parser.add_argument(
        "--method-comparison",
        type=Path,
        default=None,
        help=(
            "Optional comparison CSV used to overlay directly measured "
            "baseline runtimes."
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

    runtime_summary = build_runtime_summary(run_summary)
    reliability_summary = build_reliability_summary(run_summary)
    baseline_runtimes = load_direct_baseline_runtimes(
        args.method_comparison
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime_csv = (
        args.output_dir / "weighted_score_runtime_summary.csv"
    )
    reliability_csv = (
        args.output_dir / "weighted_score_reliability_summary.csv"
    )

    runtime_summary.to_csv(runtime_csv, index=False)
    reliability_summary.to_csv(reliability_csv, index=False)

    plot_runtime_per_run(
        runtime_summary,
        baseline_runtimes,
        args.output_dir / "weighted_score_runtime_per_run",
    )
    plot_success_rate(
        reliability_summary,
        args.output_dir / "weighted_score_success_rate",
    )

    print("\nRuntime summary:")
    print(
        runtime_summary[
            [
                "code",
                "method_label",
                "n_runs",
                "q1_runtime",
                "median_runtime",
                "q3_runtime",
                "total_runtime",
            ]
        ].to_string(index=False)
    )

    print("\nReliability summary:")
    print(
        reliability_summary[
            [
                "code",
                "method_label",
                "target_distance",
                "n_success",
                "n_runs",
                "success_percent",
            ]
        ].to_string(index=False)
    )

    print(f"\nRuntime CSV:     {runtime_csv}")
    print(f"Reliability CSV: {reliability_csv}")
    print(f"Figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
