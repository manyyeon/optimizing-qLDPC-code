#!/usr/bin/env python3
"""Plot speed-up from distance-first preprocessing.

The existing search scripts store per-candidate decoder runtimes in HDF5. This
script compares direct logical-error optimization against the distance-first
pipeline using cumulative decoder time, which is the dominant cost in these
experiments.

Example:
    python3 optimization/experiments/plot_distance_preproc_speedup.py \
        --code "[625,25]" \
        --baseline optimization/results/best_neighbor_search_early_stop.hdf5 \
        --preproc optimization/results/best_neighbor_search_d_first_2.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _read_group(path: Path, code: str):
    with h5py.File(path, "r") as f:
        if code not in f:
            available = ", ".join(f.keys())
            raise KeyError(f"{code!r} not found in {path}. Available groups: {available}")

        grp = f[code]
        data = {
            "ler": grp["logical_error_rates"][:].astype(float),
            "ler_std": grp["logical_error_rates_std"][:].astype(float),
            "runtime": grp["decoding_runtimes"][:].astype(float),
            "distance": grp["distances_quantum"][:].astype(float),
            "attrs": dict(grp.attrs),
        }

    runtime = np.nan_to_num(data["runtime"], nan=0.0, posinf=0.0, neginf=0.0)
    runtime[runtime < 0] = 0.0
    data["cum_decoder_time"] = np.cumsum(runtime)
    data["eval_index"] = np.arange(len(runtime))
    return data


def _best_so_far(ler: np.ndarray) -> np.ndarray:
    valid = np.isfinite(ler) & (ler > 0)
    out = np.full_like(ler, np.nan, dtype=float)
    best = np.inf
    for i, value in enumerate(ler):
        if valid[i]:
            best = min(best, value)
        if np.isfinite(best):
            out[i] = best
    return out


def _time_to_target(time: np.ndarray, best: np.ndarray, target: float) -> float:
    hit = np.where(np.isfinite(best) & (best <= target))[0]
    if len(hit) == 0:
        return np.nan
    return float(time[hit[0]])


def _last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if len(finite) else np.nan


def _default_target(baseline_best: np.ndarray, preproc_best: np.ndarray) -> float:
    baseline_final = _last_finite(baseline_best)
    preproc_final = _last_finite(preproc_best)
    if not np.isfinite(baseline_final) or not np.isfinite(preproc_final):
        raise ValueError("Could not infer a default target LER from the input files.")
    return max(baseline_final, preproc_final)


def _hours(seconds: float) -> float:
    return seconds / 3600.0

def _time_to_target_distance(times, distances, target):
    """Return first time when best-so-far distance reaches target."""
    times, best_dist = _best_so_far(times, distances, minimize=False)
    hit = np.where(best_dist >= target)[0]
    if len(hit) == 0:
        return np.nan
    return float(times[hit[0]])

def plot_time_to_target_box(records, output_prefix):
    """
    records: list of dicts with keys:
        code, method, seed, target, time_h
    """
    methods = sorted(set(r["method"] for r in records))
    data = []

    for method in methods:
        vals = [
            r["time_h"]
            for r in records
            if r["method"] == method and np.isfinite(r["time_h"])
        ]
        data.append(vals)

    fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
    ax.boxplot(data, tick_labels=methods, showmeans=True)
    ax.set_ylabel("Time to target distance (hours)")
    ax.set_title("Runtime benefit of distance preprocessing")
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)


def make_plot(
    baseline_path: Path,
    preproc_path: Path,
    code: str,
    output_prefix: Path,
    target_ler: float | None,
):
    baseline = _read_group(baseline_path, code)
    preproc = _read_group(preproc_path, code)

    baseline_best = _best_so_far(baseline["ler"])
    preproc_best = _best_so_far(preproc["ler"])

    target = _default_target(baseline_best, preproc_best) if target_ler is None else target_ler

    baseline_t = baseline["cum_decoder_time"]
    preproc_t = preproc["cum_decoder_time"]

    baseline_t_target = _time_to_target(baseline_t, baseline_best, target)
    preproc_t_target = _time_to_target(preproc_t, preproc_best, target)

    preproc_mc_start = np.where(np.isfinite(preproc["ler"]) & (preproc["ler"] > 0))[0]
    preproc_transition_idx = int(preproc_mc_start[0]) if len(preproc_mc_start) else None
    preproc_transition_dist = (
        np.nanmax(preproc["distance"][: preproc_transition_idx + 1])
        if preproc_transition_idx is not None
        else np.nanmax(preproc["distance"])
    )

    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[2.1, 1.0])
    ax_ler = fig.add_subplot(gs[0, :])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    colors = {
        "baseline": "#4C78A8",
        "preproc": "#F58518",
        "distance": "#54A24B",
    }

    ax_ler.step(
        _hours(baseline_t),
        baseline_best,
        where="post",
        color=colors["baseline"],
        lw=2.0,
        label="Direct LER optimization",
    )
    ax_ler.step(
        _hours(preproc_t),
        preproc_best,
        where="post",
        color=colors["preproc"],
        lw=2.0,
        label="Distance preproc. + LER optimization",
    )
    ax_ler.axhline(target, color="0.35", lw=1.0, ls=":", label=f"target $p_L$ = {target:.3g}")
    ax_ler.set_yscale("log")
    ax_ler.set_ylabel("Best logical error rate so far")
    ax_ler.set_xlabel("Cumulative decoder time (hours)")
    ax_ler.legend(frameon=False, fontsize=9)
    ax_ler.grid(True, which="both", alpha=0.25)
    ax_ler.set_title(f"{code}: speed-up from distance preprocessing")

    ax_dist.step(
        preproc["eval_index"],
        np.maximum.accumulate(np.nan_to_num(preproc["distance"], nan=0.0)),
        where="post",
        color=colors["distance"],
        lw=2.0,
    )
    if preproc_transition_idx is not None:
        ax_dist.axvline(preproc_transition_idx, color="0.25", lw=1.0, ls="--")
        ax_dist.text(
            preproc_transition_idx,
            preproc_transition_dist,
            " start LER\n optimization",
            ha="left",
            va="bottom",
            fontsize=8,
        )
    ax_dist.set_xlabel("Distance-preprocessing candidate index")
    ax_dist.set_ylabel("Best distance")
    ax_dist.grid(True, alpha=0.25)

    bar_values = [_hours(baseline_t_target), _hours(preproc_t_target)]
    ax_bar.bar(
        [0, 1],
        bar_values,
        color=[colors["baseline"], colors["preproc"]],
        width=0.65,
    )
    ax_bar.set_xticks([0, 1], ["Direct", "Two-step"], rotation=20, ha="right")
    ax_bar.set_ylabel("Hours to target")
    ax_bar.grid(True, axis="y", alpha=0.25)

    if np.isfinite(baseline_t_target) and np.isfinite(preproc_t_target) and preproc_t_target > 0:
        speedup = baseline_t_target / preproc_t_target
        ax_bar.text(
            0.5,
            max(bar_values) * 1.03,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)

    print(f"Code: {code}")
    print(f"Target LER: {target:.6g}")
    print(f"Direct time to target: {_hours(baseline_t_target):.3f} h")
    print(f"Two-step time to target: {_hours(preproc_t_target):.3f} h")
    if np.isfinite(baseline_t_target) and np.isfinite(preproc_t_target) and preproc_t_target > 0:
        print(f"Speed-up: {baseline_t_target / preproc_t_target:.2f}x")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="[625,25]", help="HDF5 group/code name")
    parser.add_argument(
        "--baseline",
        default="optimization/results/best_neighbor_search_early_stop.hdf5",
        type=Path,
        help="Direct logical-error optimization HDF5 file",
    )
    parser.add_argument(
        "--preproc",
        default="optimization/results/best_neighbor_search_d_first_2.hdf5",
        type=Path,
        help="Distance-first preprocessing HDF5 file",
    )
    parser.add_argument(
        "--target-ler",
        default=None,
        type=float,
        help="Target logical error rate. Default: best target reached by both methods.",
    )
    parser.add_argument(
        "--output-prefix",
        default="figures/distance_preprocessing_speedup",
        type=Path,
        help="Output path without extension",
    )
    args = parser.parse_args()

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    make_plot(
        baseline_path=args.baseline,
        preproc_path=args.preproc,
        code=args.code,
        output_prefix=args.output_prefix,
        target_ler=args.target_ler,
    )


if __name__ == "__main__":
    main()
