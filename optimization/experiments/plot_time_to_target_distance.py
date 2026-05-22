#!/usr/bin/env python3
"""Plot distance-preprocessing plus greedy LER optimization runtime traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_TARGETS = {
    "[625,25]": 8,
    "[1225,49]": 10,
    "[1600,64]": 10,
    "[2025,81]": 12,
}


def _read_group(path: Path, code: str):
    with h5py.File(path, "r") as f:
        if code not in f:
            raise KeyError(f"{code!r} not found in {path}. Available: {list(f.keys())}")
        grp = f[code]
        data = {
            "ler": grp["logical_error_rates"][:].astype(float),
            "std": grp["logical_error_rates_std"][:].astype(float),
            "distance": grp["distances_quantum"][:].astype(float),
            "runtime": grp["decoding_runtimes"][:].astype(float),
            "attrs": dict(grp.attrs),
        }
        if "wall_times" in grp:
            data["wall_times"] = grp["wall_times"][:].astype(float)
        else:
            data["wall_times"] = np.cumsum(np.nan_to_num(data["runtime"], nan=0.0))
        if "ler_wall_times" in grp:
            data["ler_wall_times"] = grp["ler_wall_times"][:].astype(float)
        else:
            data["ler_wall_times"] = data["wall_times"]
    return data

def _stage_total_time(data) -> float:
    """Use stored total_runtime if available; otherwise use max wall_time."""
    total = data["attrs"].get("total_runtime", np.nan)
    if np.isfinite(total):
        return float(total)

    times = data.get("wall_times", np.array([]))
    if len(times) == 0:
        return 0.0
    return float(np.nanmax(times))

def _distance_points(data, offset: float = 0.0):
    valid = np.isfinite(data["distance"]) & np.isfinite(data["wall_times"])
    times = data["wall_times"][valid] + offset
    distances = data["distance"][valid]
    return times, distances


def _best_so_far(times, values, maximize: bool = True):
    """Return times sorted with best-so-far values."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(times) & np.isfinite(values)
    times = times[valid]
    values = values[valid]

    if len(times) == 0:
        return np.array([]), np.array([])

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    best_values = np.empty_like(values, dtype=float)
    best = -np.inf if maximize else np.inf

    for i, value in enumerate(values):
        best = max(best, value) if maximize else min(best, value)
        best_values[i] = best

    return times, best_values


def _time_to_target_distance(times, distances, target_distance: float):
    """First time when best-so-far distance reaches target_distance."""
    times, best_dist = _best_so_far(times, distances, maximize=True)
    hit = np.where(best_dist >= target_distance)[0]

    if len(hit) == 0:
        return np.nan

    return float(times[hit[0]])

def _hours(seconds):
    return np.asarray(seconds, dtype=float) / 3600.0

def _valid_ler_points(data, offset=0.0, zero_floor=None):
    valid = np.isfinite(data["ler"]) & (data["ler"] > 0)
    if zero_floor is not None:
        valid = np.isfinite(data["ler"]) & (data["ler"] >= 0)
    times = data["ler_wall_times"][valid]
    fallback = data["wall_times"][valid]
    times = np.where(np.isfinite(times), times, fallback)
    values = data["ler"][valid]
    if zero_floor is not None:
        values = np.maximum(values, zero_floor)
    return times + offset, values


def make_plot(
    baseline_path: Path,
    stage1_path: Path,
    stage2_path: Path,
    code: str,
    target_distance: float,
    output_prefix: Path,
    zero_floor: float | None = None,
):
    baseline = _read_group(baseline_path, code)
    stage1 = _read_group(stage1_path, code)
    stage2 = _read_group(stage2_path, code)

    stage1_total = _stage_total_time(stage1)

    # -------------------------
    # Direct baseline timelines
    # -------------------------
    base_ler_t, base_ler = _valid_ler_points(
        baseline,
        offset=0.0,
        zero_floor=zero_floor,
    )
    base_ler_t_best, base_ler_best = _best_so_far(
        base_ler_t,
        base_ler,
        maximize=False,
    )

    base_dist_t, base_dist = _distance_points(baseline, offset=0.0)
    base_dist_t_best, base_dist_best = _best_so_far(
        base_dist_t,
        base_dist,
        maximize=True,
    )

    base_t_target = _time_to_target_distance(
        base_dist_t,
        base_dist,
        target_distance,
    )

    # -------------------------
    # Two-stage timelines
    # -------------------------
    # Stage 1 starts at t=0.
    s1_ler_t, s1_ler = _valid_ler_points(
        stage1,
        offset=0.0,
        zero_floor=zero_floor,
    )
    s1_dist_t, s1_dist = _distance_points(stage1, offset=0.0)

    # Stage 2 starts after Stage 1 finishes.
    s2_ler_t, s2_ler = _valid_ler_points(
        stage2,
        offset=stage1_total,
        zero_floor=zero_floor,
    )
    s2_dist_t, s2_dist = _distance_points(
        stage2,
        offset=stage1_total,
    )

    two_ler_t = np.concatenate([s1_ler_t, s2_ler_t])
    two_ler = np.concatenate([s1_ler, s2_ler])
    two_ler_t_best, two_ler_best = _best_so_far(
        two_ler_t,
        two_ler,
        maximize=False,
    )

    two_dist_t = np.concatenate([s1_dist_t, s2_dist_t])
    two_dist = np.concatenate([s1_dist, s2_dist])
    two_dist_t_best, two_dist_best = _best_so_far(
        two_dist_t,
        two_dist,
        maximize=True,
    )

    two_t_target = _time_to_target_distance(
        two_dist_t,
        two_dist,
        target_distance,
    )

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=(9.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.5, 1.0],
        height_ratios=[1.0, 1.0],
    )

    ax_ler = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[1, 0], sharex=ax_ler)
    ax_bar = fig.add_subplot(gs[:, 1])

    # Panel A: best LER vs wall-clock time
    ax_ler.step(
        _hours(base_ler_t_best),
        base_ler_best,
        where="post",
        lw=2.0,
        label="Direct LER optimization",
    )
    ax_ler.step(
        _hours(two_ler_t_best),
        two_ler_best,
        where="post",
        lw=2.0,
        label="Distance preproc. + LER",
    )

    ax_ler.axvline(
        stage1_total / 3600.0,
        color="0.35",
        ls="--",
        lw=1.0,
    )

    ax_ler.set_xscale("log")
    ax_ler.set_yscale("log")
    ax_ler.set_ylabel("Best logical error rate")
    ax_ler.set_title(f"[{code}]: runtime traces with distance preprocessing")
    ax_ler.grid(True, which="both", alpha=0.25)
    ax_ler.legend(frameon=False, fontsize=8)

    if zero_floor is not None:
        ax_ler.text(
            0.01,
            0.05,
            f"zero estimates plotted at {zero_floor:g}",
            transform=ax_ler.transAxes,
            fontsize=8,
            color="0.35",
        )

    # Panel B: best distance vs wall-clock time
    ax_dist.step(
        _hours(base_dist_t_best),
        base_dist_best,
        where="post",
        lw=2.0,
        label="Direct best distance",
    )
    ax_dist.step(
        _hours(two_dist_t_best),
        two_dist_best,
        where="post",
        lw=2.0,
        label="Two-stage best distance",
    )

    ax_dist.axhline(
        target_distance,
        color="0.35",
        ls=":",
        lw=1.2,
        label=f"target $d_Q={target_distance:g}$",
    )
    ax_dist.axvline(
        stage1_total / 3600.0,
        color="0.35",
        ls="--",
        lw=1.0,
    )

    ax_dist.text(
        stage1_total / 3600.0,
        ax_dist.get_ylim()[1],
        " stage 2 starts",
        ha="left",
        va="top",
        fontsize=8,
    )
    
    ax_dist.set_xscale("log")
    ax_dist.set_xlabel("Wall-clock runtime (hours)")
    ax_dist.set_ylabel("Best distance")
    ax_dist.grid(True, alpha=0.25)
    ax_dist.legend(frameon=False, fontsize=8)

    # Panel C: time to target distance
    labels = ["Direct", "Two-stage"]
    bar_values = [
        _hours(base_t_target),
        _hours(two_t_target),
    ]

    ax_bar.set_yscale("log")
    ax_bar.bar(labels, bar_values)
    ax_bar.set_ylabel("Time to target distance (hours)")
    ax_bar.set_title(f"Time to $d_Q \\geq {target_distance:g}$")
    ax_bar.grid(True, axis="y", alpha=0.25)

    if np.isfinite(base_t_target) and np.isfinite(two_t_target) and two_t_target > 0:
        speedup = base_t_target / two_t_target
        ax_bar.text(
            0.5,
            max(bar_values) * 1.03,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    # Save
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)

    # Print summary
    print(f"Code: {code}")
    print(f"Target distance: {target_distance:g}")
    print(f"Stage 1 runtime: {stage1_total:.2f} s")
    print(f"Direct time to target distance: {_hours(base_t_target):.6f} h")
    print(f"Two-stage time to target distance: {_hours(two_t_target):.6f} h")

    if np.isfinite(base_t_target) and np.isfinite(two_t_target) and two_t_target > 0:
        print(f"Distance-target speed-up: {base_t_target / two_t_target:.2f}x")

    if len(base_ler_best):
        print(f"Final direct best LER: {base_ler_best[-1]:.6g}")
    if len(two_ler_best):
        print(f"Final two-stage best LER: {two_ler_best[-1]:.6g}")
    if len(base_dist_best):
        print(f"Final direct best distance: {base_dist_best[-1]:.0f}")
    if len(two_dist_best):
        print(f"Final two-stage best distance: {two_dist_best[-1]:.0f}")

    if not np.isfinite(base_t_target):
        print("Direct LER optimization did not reach the target distance.")
    if not np.isfinite(two_t_target):
        print("Two-stage optimization did not reach the target distance.")

    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="[625,25]")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--stage2", required=True, type=Path)
    parser.add_argument("--target-distance", default=None, type=float)
    parser.add_argument(
        "--output-prefix",
        default="figures/time_to_target_distance",
        type=Path,
    )
    parser.add_argument(
    "--zero-floor",
    default=None,
    type=float,
    help="Plot zero LER estimates at this positive floor for log-scale plots.",
    )
    args = parser.parse_args()
    

    if args.target_distance is None:
        if args.code not in DEFAULT_TARGETS:
            raise ValueError(
                f"No default target distance for {args.code}. "
                "Please provide --target-distance."
            )
        target_distance = DEFAULT_TARGETS[args.code]
    else:
        target_distance = args.target_distance

    make_plot(
        baseline_path=args.baseline,
        stage1_path=args.stage1,
        stage2_path=args.stage2,
        code=args.code,
        target_distance=target_distance,
        output_prefix=args.output_prefix,
        zero_floor=args.zero_floor,
    )


if __name__ == "__main__":
    main()