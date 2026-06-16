#!/usr/bin/env python3
"""Analyze repeated greedy/beam qLDPC search runs stored in HDF5 files.

Expected filenames (default pattern):
    logical_guided_greedy_absolute_score_gamma01_slack1_C0_run1.hdf5
    logical_guided_beam_absolute_score_gamma01_slack1_C0_run1.hdf5
    ... C0--C3, run1--run10

The script is intentionally tolerant of small HDF5 schema differences. It searches
for common dataset names such as distance, low_weight_scores, ler, runtime, step,
accepted, selected_beam_rank, precision_selected, and final_best.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"logical_guided_(?P<method>greedy|beam)_.*?_C(?P<C>\d+)_run(?P<run>\d+)\.hdf5$"
)

# Exact basename aliases, in priority order.
ALIASES: dict[str, tuple[str, ...]] = {
    "distance": (
        "distances_quantum",
        "quantum_distance",
        "hgp_distance",
        "logical_distance",
        "distance",
        "distances",
        "d_q",
        "score_d_q",
    ),
    "score": (
        "low_weight_scores",
        "low_weight_score",
        "weight_score",
        "logical_score",
        "score",
        "scores",
    ),
    "ler": (
        "prec_logical_error_rates",
        "logical_error_rates",
        "logical_error_rate",
        "ler",
    ),
    "ler_std": (
        "prec_logical_error_rates_std",
        "logical_error_rates_std",
        "logical_error_rate_std",
        "ler_std",
        "std",
    ),
    "runtime": (
        "prec_decoding_runtimes",
        "decoding_runtimes",
        "runtime",
        "runtimes",
        "elapsed_time",
        "elapsed_seconds",
        "time_seconds",
    ),
    "step": (
        "search_step",
        "step",
        "steps",
        "iteration",
        "iterations",
    ),
    "trial": (
        "search_trial",
        "trial",
        "trials",
    ),
    "accepted": (
        "accepted",
        "is_accepted",
    ),
    "final_best": (
        "final_best",
        "is_final_best",
    ),
    "precision_selected": (
        "precision_selected",
        "selected_for_precision",
        "is_precision_selected",
    ),
    "selected_beam_rank": (
        "selected_beam_rank",
        "beam_rank",
        "selected_rank",
    ),
    "parent_idx": (
        "parent_idx",
        "parent_index",
    ),
}

CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "beam_width": ("beam_width", "beam-width", "beamwidth", "bw"),
    "score_gamma": ("score_gamma", "gamma"),
    "score_max_weight": ("score_max_weight", "max_weight", "maxw"),
    "score_max_top": ("score_max_top", "max_top", "scoretop"),
    "logical_max_comb_order": (
        "logical_max_comb_order",
        "max_comb_order",
        "comb_order",
    ),
    "precision_budget": ("prec_budget", "precision_budget"),
    "total_runtime": ("total_runtime",),
}


@dataclass
class RunTrace:
    method: str
    C: int
    run: int
    code_family: str
    group_path: str
    frame: pd.DataFrame


def safe_array(dataset: h5py.Dataset) -> np.ndarray | None:
    """Read a dataset and return a numeric/boolean ndarray when possible."""
    try:
        arr = np.asarray(dataset[()])
    except Exception as exc:  # pragma: no cover - defensive for malformed files
        warnings.warn(f"Could not read dataset {dataset.name}: {exc}")
        return None

    if arr.ndim == 0:
        return arr.reshape(1)

    # Most record-wise datasets should be one-dimensional. A trailing singleton
    # dimension is harmless and common with append-style HDF5 writers.
    if arr.ndim > 1 and int(np.prod(arr.shape[1:])) == 1:
        return arr.reshape(arr.shape[0])

    return arr


def dataset_map(group: h5py.Group) -> dict[str, list[h5py.Dataset]]:
    """Map lowercase dataset basenames to all matching datasets below group."""
    result: dict[str, list[h5py.Dataset]] = {}

    def visitor(_: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            key = obj.name.rsplit("/", 1)[-1].lower()
            result.setdefault(key, []).append(obj)

    group.visititems(visitor)
    return result


def choose_dataset(
    mapping: dict[str, list[h5py.Dataset]], aliases: Iterable[str]
) -> h5py.Dataset | None:
    """Choose the shallowest dataset matching the first available alias."""
    for alias in aliases:
        matches = mapping.get(alias.lower(), [])
        if matches:
            return sorted(matches, key=lambda d: (d.name.count("/"), len(d.name)))[0]
    return None


def one_dimensional_record_array(
    mapping: dict[str, list[h5py.Dataset]], key: str
) -> np.ndarray | None:
    ds = choose_dataset(mapping, ALIASES[key])
    if ds is None:
        return None
    arr = safe_array(ds)
    if arr is None:
        return None
    if arr.ndim == 1:
        return arr

    # Avoid silently reducing genuinely multidimensional diagnostic arrays.
    warnings.warn(
        f"Ignoring multidimensional dataset {ds.name} with shape {arr.shape} "
        f"for scalar field '{key}'."
    )
    return None


def normalized_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    arr = np.asarray(value)
    if arr.size == 1:
        item = arr.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        if isinstance(item, np.generic):
            return item.item()
        return item
    return None


def find_config_value(group: h5py.Group, aliases: Iterable[str]) -> Any:
    """Search attrs and scalar datasets recursively for a configuration value."""
    alias_set = {a.lower().replace("-", "_") for a in aliases}

    # Search group and descendant attributes.
    objects: list[h5py.Group | h5py.Dataset] = [group]

    def collect(_: str, obj: h5py.Group | h5py.Dataset) -> None:
        objects.append(obj)

    group.visititems(collect)
    for obj in objects:
        for key, value in obj.attrs.items():
            normalized_key = key.lower().replace("-", "_")
            if normalized_key in alias_set:
                scalar = normalized_scalar(value)
                if scalar is not None:
                    return scalar

    mapping = dataset_map(group)
    for alias in aliases:
        ds = choose_dataset(mapping, (alias, alias.replace("-", "_")))
        if ds is not None:
            arr = safe_array(ds)
            if arr is not None and arr.size == 1:
                return normalized_scalar(arr)
    return np.nan


def find_run_groups(h5: h5py.File) -> list[h5py.Group]:
    """Locate logical-guided run groups, with a dataset-based fallback."""
    named: list[h5py.Group] = []
    fallback: list[h5py.Group] = []

    def visitor(_: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Group):
            return
        base = obj.name.rsplit("/", 1)[-1].lower()
        direct_names = {k.lower() for k in obj.keys()}
        has_core = bool(
            direct_names
            & {
                "distance",
                "distances",
                "low_weight_scores",
                "low_weight_score",
                "step",
                "steps",
                "final_best",
            }
        )
        if base.startswith("logical_guided"):
            named.append(obj)
        elif has_core:
            fallback.append(obj)

    h5.visititems(visitor)

    candidates = named if named else fallback
    # Remove groups nested inside another selected group to avoid duplicate reads.
    candidates = sorted(candidates, key=lambda g: g.name.count("/"))
    selected: list[h5py.Group] = []
    for group in candidates:
        if not any(group.name.startswith(parent.name.rstrip("/") + "/") for parent in selected):
            selected.append(group)
    return selected


def infer_record_count(arrays: dict[str, np.ndarray | None]) -> int:
    preferred = (
        "distance",
        "score",
        "step",
        "accepted",
        "selected_beam_rank",
        "final_best",
        "ler",
    )
    lengths = [len(arrays[k]) for k in preferred if arrays.get(k) is not None]
    if not lengths:
        return 0
    # The mode is safer than max when one dataset is accidentally scalar metadata.
    values, counts = np.unique(lengths, return_counts=True)
    return int(values[np.argmax(counts)])


def aligned_series(arr: np.ndarray | None, n: int, name: str) -> pd.Series:
    if arr is None:
        return pd.Series(np.full(n, np.nan), name=name)
    if len(arr) == n:
        return pd.Series(arr, name=name)
    if len(arr) == 1:
        return pd.Series(np.repeat(arr[0], n), name=name)
    warnings.warn(
        f"Dataset '{name}' has length {len(arr)}, but inferred record count is {n}; "
        "filling this field with NaN."
    )
    return pd.Series(np.full(n, np.nan), name=name)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def to_bool_mask(series: pd.Series) -> np.ndarray:
    if series.isna().all():
        return np.zeros(len(series), dtype=bool)
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).to_numpy() != 0
    lowered = series.astype(str).str.lower()
    return lowered.isin({"true", "t", "yes", "y", "1"}).to_numpy()


def build_frame(group: h5py.Group) -> pd.DataFrame:
    mapping = dataset_map(group)
    arrays = {key: one_dimensional_record_array(
        mapping, key) for key in ALIASES}
    n = infer_record_count(arrays)
    if n == 0:
        return pd.DataFrame()

    frame = pd.DataFrame({"record_index": np.arange(n, dtype=int)})
    for key, arr in arrays.items():
        frame[key] = aligned_series(arr, n, key)

    numeric_cols = [
        "distance",
        "score",
        "ler",
        "ler_std",
        "runtime",
        "step",
        "trial",
        "selected_beam_rank",
        "parent_idx",
    ]
    for col in numeric_cols:
        frame[col] = to_numeric(frame[col])

    if frame["step"].isna().all():
        frame["step"] = frame["record_index"]
    return frame


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def select_final_index(frame: pd.DataFrame) -> tuple[int, str]:
    """Select the final code using explicit markers first, then robust fallbacks."""
    for column, label in (
        ("final_best", "final_best marker"),
        ("precision_selected", "precision_selected marker"),
    ):
        mask = to_bool_mask(frame[column])
        if mask.any():
            return int(frame.loc[mask, "record_index"].iloc[-1]), label

    # For beam traces, choose the smallest selected rank in the latest step.
    if frame["selected_beam_rank"].notna().any():
        latest_step = frame["step"].max()
        candidates = frame[frame["step"] == latest_step].copy()
        candidates = candidates[candidates["selected_beam_rank"].notna()]
        if not candidates.empty:
            idx = candidates.sort_values(
                ["selected_beam_rank", "score", "record_index"],
                ascending=[True, True, False],
                na_position="last",
            )["record_index"].iloc[0]
            return int(idx), "lowest beam rank at latest step"

    accepted_mask = to_bool_mask(frame["accepted"])
    if accepted_mask.any():
        accepted = frame.loc[accepted_mask]
        latest_step = accepted["step"].max()
        idx = accepted.loc[accepted["step"] ==
                           latest_step, "record_index"].iloc[-1]
        return int(idx), "last accepted state"

    latest_step = frame["step"].max()
    idx = frame.loc[frame["step"] == latest_step, "record_index"].iloc[-1]
    return int(idx), "last record at latest step"


def select_best_index(frame: pd.DataFrame) -> int:
    """Select best discovered code: maximum distance, then minimum score."""
    candidates = frame.copy()
    if candidates["distance"].notna().any():
        max_d = candidates["distance"].max()
        candidates = candidates[candidates["distance"] == max_d]
    if candidates["score"].notna().any():
        min_score = candidates["score"].min()
        candidates = candidates[candidates["score"] == min_score]
    return int(candidates["record_index"].iloc[-1])


def row_value(frame: pd.DataFrame, index: int, column: str) -> float:
    value = frame.loc[frame["record_index"] == index, column]
    if value.empty:
        return np.nan
    return finite_or_nan(value.iloc[0])


def runtime_seconds(frame: pd.DataFrame, final_index: int) -> float:
    values = frame["runtime"].dropna()
    if values.empty:
        return np.nan
    # If runtime is cumulative, max is correct. If it is per-record, sum is more
    # appropriate. Detect monotonicity before deciding.
    arr = values.to_numpy(dtype=float)
    if len(arr) == 1:
        return finite_or_nan(arr[0])
    if np.all(np.diff(arr) >= -1e-12):
        return finite_or_nan(np.max(arr))
    return finite_or_nan(np.sum(arr[arr >= 0]))


def code_family_from_group(group_path: str) -> str:
    parts = [part for part in group_path.split("/") if part]
    return parts[0] if parts else "unknown"


def make_trace(
    frame: pd.DataFrame,
    method: str,
    C: int,
    run: int,
    code_family: str,
    group_path: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    ordered = frame.sort_values(["step", "record_index"], kind="stable")
    best_d = -np.inf
    best_score_at_d = np.nan
    global_min_score = np.inf
    rows: list[dict[str, Any]] = []
    evaluated = 0

    for step, chunk in ordered.groupby("step", sort=True, dropna=False):
        for _, row in chunk.iterrows():
            evaluated += 1
            d = finite_or_nan(row["distance"])
            s = finite_or_nan(row["score"])

            if math.isfinite(s):
                global_min_score = min(global_min_score, s)

            if math.isfinite(d):
                if d > best_d:
                    best_d = d
                    best_score_at_d = s
                elif d == best_d and math.isfinite(s):
                    if not math.isfinite(best_score_at_d) or s < best_score_at_d:
                        best_score_at_d = s

        rows.append(
            {
                "method": method,
                "C": C,
                "run": run,
                "code_family": code_family,
                "group_path": group_path,
                "step": finite_or_nan(step),
                "n_evaluated": evaluated,
                "best_distance_so_far": best_d if math.isfinite(best_d) else np.nan,
                "best_score_at_best_distance_so_far": best_score_at_d,
                "global_min_score_so_far": (
                    global_min_score if math.isfinite(
                        global_min_score) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def quantile(series: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.quantile(q)) if not numeric.empty else np.nan


def aggregate_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # A common target for a fair greedy-vs-beam comparison: the largest distance
    # reached by either method for the same C/code family.
    comparison_targets: dict[tuple[int, str], float] = {}
    for (C, code_family), family_group in run_df.groupby(
        ["C", "code_family"], dropna=False
    ):
        values = pd.to_numeric(family_group["best_distance"], errors="coerce")
        comparison_targets[(C, code_family)] = (
            float(values.max()) if values.notna().any() else np.nan
        )

    for (method, C, code_family), group in run_df.groupby(
        ["method", "C", "code_family"], dropna=False
    ):
        best_distances = pd.to_numeric(group["best_distance"], errors="coerce")
        final_distances = pd.to_numeric(
            group["final_distance"], errors="coerce")
        method_best = best_distances.max() if best_distances.notna().any() else np.nan
        method_success_count = (
            int((best_distances == method_best).sum()
                ) if math.isfinite(method_best) else 0
        )
        method_success_rate = method_success_count / \
            len(group) if len(group) else np.nan

        comparison_target = comparison_targets[(C, code_family)]
        comparison_success_count = (
            int((best_distances >= comparison_target).sum())
            if math.isfinite(comparison_target)
            else 0
        )
        comparison_success_rate = (
            comparison_success_count / len(group) if len(group) else np.nan
        )

        # Runs that discovered the method's maximum distance.
        at_best_distance = (
            group[best_distances == method_best]
            if math.isfinite(method_best)
            else group.iloc[0:0]
        )

        best_score_at_method_best = (
            pd.to_numeric(
                at_best_distance["best_score"],
                errors="coerce",
            ).min()
            if not at_best_distance.empty
            else np.nan
        )

        # Use the final selected code for LER because final selection is
        # distance first, then precision LER.
        at_final_best_distance = (
            group[final_distances == method_best]
            if math.isfinite(method_best)
            else group.iloc[0:0]
        )

        best_ler_at_method_best = (
            pd.to_numeric(
                at_final_best_distance["final_ler"],
                errors="coerce",
            ).min()
            if not at_final_best_distance.empty
            else np.nan
        )

        runtime = pd.to_numeric(group["runtime_seconds"], errors="coerce")
        rows.append(
            {
                "method": method,
                "C": C,
                "code_family": code_family,
                "n_runs": len(group),
                "method_best_distance": method_best,
                "runs_reaching_method_best": method_success_count,
                "success_rate_at_method_best": method_success_rate,
                "comparison_target_distance": comparison_target,
                "runs_reaching_comparison_target": comparison_success_count,
                "success_rate_at_comparison_target": comparison_success_rate,
                "median_best_distance": quantile(best_distances, 0.5),
                "q25_best_distance": quantile(best_distances, 0.25),
                "q75_best_distance": quantile(best_distances, 0.75),
                "min_best_distance": best_distances.min(),
                "max_best_distance": best_distances.max(),
                "median_final_distance": quantile(final_distances, 0.5),
                "q25_final_distance": quantile(final_distances, 0.25),
                "q75_final_distance": quantile(final_distances, 0.75),
                "best_score_at_method_best_distance": best_score_at_method_best,
                "best_ler_at_method_best_distance": best_ler_at_method_best,
                "median_runtime_seconds": quantile(runtime, 0.5),
                "total_runtime_seconds": runtime.sum(min_count=1),
                "median_records_evaluated": quantile(group["n_records"], 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values(["C", "method", "code_family"])


def aggregate_convergence(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame()

    output: list[pd.DataFrame] = []
    metrics = (
        "best_distance_so_far",
        "best_score_at_best_distance_so_far",
        "global_min_score_so_far",
        "n_evaluated",
    )

    for (method, C, code_family), group in trace_df.groupby(
        ["method", "C", "code_family"], dropna=False
    ):
        all_steps = np.sort(group["step"].dropna().unique())
        if len(all_steps) == 0:
            continue

        per_run: dict[int, pd.DataFrame] = {}
        for run, run_group in group.groupby("run"):
            indexed = run_group.sort_values(
                "step").drop_duplicates("step", keep="last")
            indexed = indexed.set_index("step").reindex(all_steps).ffill()
            per_run[int(run)] = indexed

        for metric in metrics:
            matrix = pd.DataFrame(
                {
                    run: pd.to_numeric(df[metric], errors="coerce")
                    for run, df in per_run.items()
                },
                index=all_steps,
            )
            summary = pd.DataFrame(
                {
                    "method": method,
                    "C": C,
                    "code_family": code_family,
                    "step": all_steps,
                    "metric": metric,
                    "median": matrix.median(axis=1, skipna=True).to_numpy(),
                    "q25": matrix.quantile(0.25, axis=1).to_numpy(),
                    "q75": matrix.quantile(0.75, axis=1).to_numpy(),
                    "min": matrix.min(axis=1, skipna=True).to_numpy(),
                    "max": matrix.max(axis=1, skipna=True).to_numpy(),
                    "n_runs_available": matrix.notna().sum(axis=1).to_numpy(),
                }
            )
            output.append(summary)

    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def distance_histogram(run_df: pd.DataFrame) -> pd.DataFrame:
    data = run_df.dropna(subset=["best_distance"]).copy()
    if data.empty:
        return pd.DataFrame()
    data["best_distance"] = pd.to_numeric(
        data["best_distance"], errors="coerce")
    return (
        data.groupby(["method", "C", "code_family", "best_distance"])
        .size()
        .rename("run_count")
        .reset_index()
        .sort_values(["C", "method", "best_distance"])
    )


def _code_label(data: pd.DataFrame, C: int) -> str:
    """Return a concise legend label such as 'C1 [1225,49]'."""
    families = (
        data.loc[data["C"] == C, "code_family"]
        .dropna()
        .astype(str)
        .unique()
    )
    if len(families) == 1 and families[0] != "unknown":
        return f"C{C} {families[0]}"
    return f"C{C}"


def plot_best_distance_by_run(run_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot C0--C3 and both methods in one figure."""
    if run_df.empty:
        return

    plt.figure(figsize=(10.0, 5.8))

    linestyles = {
        "greedy": "--",
        "beam": "-",
    }
    markers = {
        "greedy": "s",
        "beam": "o",
    }

    # Matplotlib chooses one default color per C. Greedy and beam for the
    # same C reuse that color and are distinguished by line style/marker.
    color_by_C: dict[int, str] = {}

    C_values = sorted(
        pd.to_numeric(run_df["C"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )

    for C in C_values:
        C_group = run_df[run_df["C"] == C]
        family_label = _code_label(run_df, C)

        for method in ("greedy", "beam"):
            method_group = C_group[C_group["method"]
                                   == method].sort_values("run")
            if method_group.empty:
                continue

            line = plt.plot(
                method_group["run"],
                method_group["best_distance"],
                linestyle=linestyles.get(method, "-"),
                marker=markers.get(method, "o"),
                linewidth=1.8,
                markersize=5,
                label=f"{family_label} {method.capitalize()}",
                color=color_by_C.get(C),
            )[0]
            color_by_C.setdefault(C, line.get_color())

    plt.xlabel("Independent run")
    plt.ylabel("Best distance discovered")
    plt.title("Best distance across repeated runs")
    plt.xticks(
        sorted(
            pd.to_numeric(run_df["run"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
    )
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "best_distance_by_run_all_codes.png", dpi=220)
    plt.close()


def plot_convergence(convergence_df: pd.DataFrame, output_dir: Path) -> None:
    """Create one combined C0--C3 figure for each convergence metric."""
    if convergence_df.empty:
        return

    plot_specs = {
        "best_distance_so_far": (
            "Best distance found so far",
            "convergence_distance_all_codes",
            False,
        ),
        "best_score_at_best_distance_so_far": (
            "Best score at current best distance",
            "convergence_score_at_best_distance_all_codes",
            True,
        ),
        "global_min_score_so_far": (
            "Minimum score found so far",
            "convergence_min_score_all_codes",
            True,
        ),
    }

    linestyles = {
        "greedy": "--",
        "beam": "-",
    }

    for metric, metric_group in convergence_df.groupby("metric"):
        if metric not in plot_specs:
            continue

        ylabel, filename, use_log_scale = plot_specs[metric]
        plt.figure(figsize=(10.0, 5.8))
        color_by_C: dict[int, str] = {}

        C_values = sorted(
            pd.to_numeric(metric_group["C"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )

        for C in C_values:
            C_group = metric_group[metric_group["C"] == C]
            family_label = _code_label(metric_group, C)

            for method in ("greedy", "beam"):
                method_group = C_group[C_group["method"]
                                       == method].sort_values("step")
                if method_group.empty:
                    continue

                x = method_group["step"].to_numpy(dtype=float)
                median = method_group["median"].to_numpy(dtype=float)
                q25 = method_group["q25"].to_numpy(dtype=float)
                q75 = method_group["q75"].to_numpy(dtype=float)

                # Log plots cannot display nonpositive values. Convert those
                # entries to NaN rather than distorting or failing the plot.
                if use_log_scale:
                    median = np.where(median > 0, median, np.nan)
                    q25 = np.where(q25 > 0, q25, np.nan)
                    q75 = np.where(q75 > 0, q75, np.nan)

                line = plt.plot(
                    x,
                    median,
                    linestyle=linestyles.get(method, "-"),
                    linewidth=1.8,
                    label=f"{family_label} {method.capitalize()}",
                    color=color_by_C.get(C),
                )[0]
                color_by_C.setdefault(C, line.get_color())

                plt.fill_between(
                    x,
                    q25,
                    q75,
                    color=color_by_C[C],
                    alpha=0.08,
                )

        plt.xlabel("Search step")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} across all code families")
        if use_log_scale:
            plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename}.png", dpi=220)
        plt.close()


def plot_success_curves(
    trace_df: pd.DataFrame,
    run_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot C0--C3 success probabilities together in one figure.

    Each code family uses its own target distance: the largest distance reached
    by either method among the repeated runs for that C value.
    """
    if trace_df.empty or run_df.empty:
        return

    plt.figure(figsize=(10.0, 5.8))

    linestyles = {
        "greedy": "--",
        "beam": "-",
    }
    color_by_C: dict[int, str] = {}

    C_values = sorted(
        pd.to_numeric(run_df["C"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )

    for C in C_values:
        C_runs = run_df[run_df["C"] == C]
        C_trace = trace_df[trace_df["C"] == C]
        family_label = _code_label(run_df, C)

        target_distance = pd.to_numeric(
            C_runs["best_distance"], errors="coerce"
        ).max()
        if not math.isfinite(target_distance):
            continue

        all_steps = np.sort(C_trace["step"].dropna().unique())
        if len(all_steps) == 0:
            continue

        for method in ("greedy", "beam"):
            method_trace = C_trace[C_trace["method"] == method]
            if method_trace.empty:
                continue

            run_curves: list[np.ndarray] = []
            for _, one_run in method_trace.groupby("run"):
                series = (
                    one_run.sort_values("step")
                    .drop_duplicates("step", keep="last")
                    .set_index("step")["best_distance_so_far"]
                    .reindex(all_steps)
                    .ffill()
                )
                reached_target = (
                    series.to_numpy(dtype=float) >= target_distance
                ).astype(float)
                run_curves.append(reached_target)

            if not run_curves:
                continue

            success_probability = np.vstack(run_curves).mean(axis=0)
            line = plt.plot(
                all_steps,
                success_probability,
                linestyle=linestyles.get(method, "-"),
                linewidth=1.8,
                label=(
                    f"{family_label} {method.capitalize()} "
                    f"(target d≥{target_distance:g})"
                ),
                color=color_by_C.get(C),
            )[0]
            color_by_C.setdefault(C, line.get_color())

    plt.xlabel("Search step")
    plt.ylabel("Fraction of runs reaching target distance")
    plt.ylim(-0.03, 1.03)
    plt.title("Success probability across all code families")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "success_probability_all_codes.png", dpi=220)
    plt.close()


def warn_missing_runs(run_df: pd.DataFrame, expected_runs: int) -> None:
    expected = set(range(1, expected_runs + 1))
    for (method, C), group in run_df.groupby(["method", "C"]):
        found = set(pd.to_numeric(
            group["run"], errors="coerce").dropna().astype(int))
        missing = sorted(expected - found)
        duplicates = sorted(
            group.loc[group["run"].duplicated(
                keep=False), "run"].astype(int).unique()
        )
        if missing:
            print(
                f"WARNING: {method}, C={C}: missing runs {missing}", file=sys.stderr)
        if duplicates:
            print(
                f"WARNING: {method}, C={C}: multiple run groups for run IDs {duplicates}",
                file=sys.stderr,
            )


def select_best_precision_ler_at_distance(
    frame: pd.DataFrame,
    target_distance: float,
) -> tuple[float, float, int]:
    """
    Return the minimum precision LER among evaluated candidates
    at target_distance.
    """
    precision_mask = to_bool_mask(frame["precision_selected"])

    candidates = frame.loc[
        precision_mask
        & frame["distance"].notna()
        & np.isclose(
            frame["distance"].to_numpy(dtype=float),
            float(target_distance),
        )
        & frame["ler"].notna()
    ].copy()

    if candidates.empty:
        return np.nan, np.nan, -1

    candidates = candidates.sort_values(
        ["ler", "score", "record_index"],
        ascending=[True, True, True],
        na_position="last",
    )

    best = candidates.iloc[0]

    return (
        finite_or_nan(best["ler"]),
        finite_or_nan(best["ler_std"]),
        int(best["record_index"]),
    )


def analyze(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search beam/, greedy/, and any other nested directories.
    files = sorted(results_dir.rglob(args.pattern))

    if not files:
        raise FileNotFoundError(
            f"No files recursively matched pattern {args.pattern!r} "
            f"under {results_dir}. "
            "Check --results-dir and --pattern."
        )

    print(f"Found {len(files)} HDF5 files under {results_dir}.")

    summary_rows: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []

    for path in files:
        match = FILENAME_RE.search(path.name)

        if not match:
            print(
                f"Skipping unrecognized filename: {path}",
                file=sys.stderr,
            )
            continue

        method = match.group("method")
        C = int(match.group("C"))
        run = int(match.group("run"))

        folder_method = path.parent.name.lower()

        if (
            folder_method in {"beam", "greedy"}
            and folder_method != method
        ):
            print(
                f"WARNING: method mismatch for {path}: "
                f"folder says {folder_method!r}, "
                f"but filename says {method!r}.",
                file=sys.stderr,
            )

        try:
            with h5py.File(path, "r") as h5:
                groups = find_run_groups(h5)

                if not groups:
                    print(
                        f"WARNING: no run group found in {path}",
                        file=sys.stderr,
                    )
                    continue

                for group in groups:
                    frame = build_frame(group)

                    if frame.empty:
                        print(
                            f"WARNING: no record-wise datasets found "
                            f"in {path}:{group.name}",
                            file=sys.stderr,
                        )
                        continue

                    final_idx, final_rule = select_final_index(frame)
                    best_idx = select_best_index(frame)
                    run_best_distance = row_value(
                        frame,
                        best_idx,
                        "distance",
                    )

                    (
                        best_precision_ler,
                        best_precision_ler_std,
                        best_precision_ler_idx,
                    ) = select_best_precision_ler_at_distance(
                        frame,
                        run_best_distance,
                    )
                    code_family = code_family_from_group(group.name)

                    config = {
                        name: find_config_value(group, aliases)
                        for name, aliases in CONFIG_ALIASES.items()
                    }

                    total_runtime = finite_or_nan(
                        config.get("total_runtime", np.nan)
                    )

                    if math.isfinite(total_runtime):
                        run_runtime_seconds = total_runtime
                        runtime_source = (
                            "run-group total_runtime attribute"
                        )
                    else:
                        run_runtime_seconds = runtime_seconds(
                            frame,
                            final_idx,
                        )
                        runtime_source = "record runtime fallback"

                    row = {
                        "file": str(path),
                        "filename": path.name,
                        "method_directory": folder_method,
                        "method": method,
                        "C": C,
                        "run": run,
                        "code_family": code_family,
                        "group_path": group.name,
                        "n_records": len(frame),
                        "n_steps": frame["step"].nunique(
                            dropna=True
                        ),
                        "final_index": final_idx,
                        "final_selection_rule": final_rule,
                        "final_distance": row_value(
                            frame, final_idx, "distance"
                        ),
                        "final_score": row_value(
                            frame, final_idx, "score"
                        ),
                        "final_ler": row_value(
                            frame, final_idx, "ler"
                        ),
                        "final_ler_std": row_value(
                            frame, final_idx, "ler_std"
                        ),
                        "best_index": best_idx,
                        "best_distance": row_value(
                            frame, best_idx, "distance"
                        ),
                        "best_score": row_value(
                            frame, best_idx, "score"
                        ),
                        "best_ler": best_precision_ler,
                        "best_ler_std": best_precision_ler_std,
                        "best_ler_index": best_precision_ler_idx,
                        "runtime_seconds": run_runtime_seconds,
                        "runtime_source": runtime_source,
                        **config,
                    }

                    summary_rows.append(row)
                    traces.append(
                        make_trace(
                            frame,
                            method,
                            C,
                            run,
                            code_family,
                            group.name,
                        )
                    )

        except OSError as exc:
            print(
                f"ERROR opening {path}: {exc}",
                file=sys.stderr,
            )
    if not summary_rows:
        raise RuntimeError(
            "Files were found, but no analyzable run groups were extracted."
        )

    run_df = pd.DataFrame(summary_rows).sort_values(
        ["C", "method", "run", "code_family", "group_path"]
    )

    trace_df = (
        pd.concat(
            [trace for trace in traces if not trace.empty],
            ignore_index=True,
        )
        if any(not trace.empty for trace in traces)
        else pd.DataFrame()
    )

    warn_missing_runs(run_df, args.expected_runs)

    greedy_bw = pd.to_numeric(
        run_df.loc[
            run_df["method"] == "greedy",
            "beam_width",
        ],
        errors="coerce",
    ).dropna()

    if not greedy_bw.empty and (greedy_bw != 1).any():
        print(
            "WARNING: At least one file named 'greedy' "
            "has beam_width != 1.",
            file=sys.stderr,
        )

    aggregate_df = aggregate_runs(run_df)
    histogram_df = distance_histogram(run_df)
    convergence_df = aggregate_convergence(trace_df)

    run_df.to_csv(
        output_dir / "run_summary.csv",
        index=False,
    )
    aggregate_df.to_csv(
        output_dir / "aggregate_summary.csv",
        index=False,
    )
    histogram_df.to_csv(
        output_dir / "distance_histogram.csv",
        index=False,
    )

    if not trace_df.empty:
        trace_df.to_csv(
            output_dir / "convergence_by_run.csv",
            index=False,
        )

    if not convergence_df.empty:
        convergence_df.to_csv(
            output_dir / "convergence_aggregate.csv",
            index=False,
        )

    plot_best_distance_by_run(run_df, output_dir)
    plot_convergence(convergence_df, output_dir)
    plot_success_curves(trace_df, run_df, output_dir)

    display_cols = [
        "method",
        "C",
        "code_family",
        "n_runs",
        "method_best_distance",
        "runs_reaching_method_best",
        "success_rate_at_method_best",
        "comparison_target_distance",
        "runs_reaching_comparison_target",
        "success_rate_at_comparison_target",
        "median_best_distance",
        "q25_best_distance",
        "q75_best_distance",
        "best_score_at_method_best_distance",
        "best_ler_at_method_best_distance",
        "median_runtime_seconds",
    ]

    print("\n=== Aggregate repeated-run summary ===")
    print(aggregate_df[display_cols].to_string(index=False))
    print(f"\nSaved CSV files and figures to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze repeated greedy and beam qLDPC HDF5 search runs."
        )
    )

    parser.add_argument(
        "--results-dir",
        default=(
            "optimization/results/"
            "logical_guided_absolute_score_gamma01_slack1_"
            "repeated_runs"
        ),
        help=(
            "Root directory containing the beam/ and greedy/ "
            "result directories."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "optimization/results/"
            "logical_guided_absolute_score_gamma01_slack1_"
            "repeated_runs/analysis"
        ),
        help="Directory for CSV summaries and PNG plots.",
    )

    parser.add_argument(
        "--pattern",
        default=(
            "logical_guided_*_absolute_score_gamma01_"
            "slack1_C*_run*.hdf5"
        ),
        help="Recursive filename pattern below --results-dir.",
    )

    parser.add_argument(
        "--expected-runs",
        type=int,
        default=10,
        help="Expected independent run IDs per method and C value.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
