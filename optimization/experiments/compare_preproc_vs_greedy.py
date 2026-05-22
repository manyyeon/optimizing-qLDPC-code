#!/usr/bin/env python3
"""Run paired experiments: preprocessing (distance-first) vs pure greedy.

Saves per-run JSON summaries under `optimization/results/compare_runs/`.
"""
import subprocess
import time
import json
import os
import h5py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "optimization" / "results" / "compare_runs"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_command(cmd, env=None):
    start = time.time()
    proc = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    end = time.time()
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "wall_time": end - start,
    }


def extract_hdf5_metrics(hdf5_path, group_name):
    if not os.path.exists(hdf5_path):
        return {}
    try:
        with h5py.File(hdf5_path, "r") as f:
            if group_name not in f:
                return {}
            grp = f[group_name]
            metrics = {}
            # common attrs
            for k, v in grp.attrs.items():
                try:
                    metrics[k] = float(v)
                except Exception:
                    metrics[k] = v

            # dataset sizes
            if "states" in grp:
                metrics["num_states"] = grp["states"].shape[0]
            return metrics
    except Exception as e:
        return {"hdf5_error": str(e)}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-C', default=0, type=int, help='Code family index')
    parser.add_argument('--seeds', default='0', help='Comma-separated seeds')
    parser.add_argument('--preproc-script', default='python3 optimization/best_neighbor_search_d_first.py -C {C} 2>&1 | tee -a "run_logs/preproc_C{C}_seed{SEED}_$(date +%Y%m%d_%H%M%S).log"')
    parser.add_argument('--greedy-script', default='python3 optimization/best_neighbor_search.py -C {C} 2>&1 | tee -a "run_logs/greedy_C{C}_seed{SEED}_$(date +%Y%m%d_%H%M%S).log"')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s != '']

    # locations of script outputs (hardcoded to current scripts)
    preproc_h5 = ROOT / 'optimization' / 'results' / 'best_neighbor_search_d_first_2.hdf5'
    greedy_h5 = ROOT / 'optimization' / 'results' / 'best_neighbor_search_5.hdf5'

    for seed in seeds:
        print(f"Running seed={seed} (C={args.C})")

        # set reproducible env var (scripts may or may not use it)
        env = os.environ.copy()
        env['SEED'] = str(seed)

        # Run preprocessing pipeline, set unique output HDF5 to avoid file locks
        preproc_out = ROOT / 'optimization' / 'results' / f"best_neighbor_search_d_first_C{args.C}_seed{seed}.hdf5"
        env_pre = env.copy()
        env_pre['OUTPUT_HDF5'] = str(preproc_out)
        cmd_pre = args.preproc_script.format(C=args.C, SEED=seed)
        res_pre = run_command(cmd_pre, env=env_pre)
        metrics_pre = extract_hdf5_metrics(str(preproc_out), '')

        # Run pure greedy baseline
        greedy_out = ROOT / 'optimization' / 'results' / f"best_neighbor_search_C{args.C}_seed{seed}.hdf5"
        env_greedy = env.copy()
        env_greedy['OUTPUT_HDF5'] = str(greedy_out)
        cmd_greedy = args.greedy_script.format(C=args.C, SEED=seed)
        res_greedy = run_command(cmd_greedy, env=env_greedy)
        metrics_greedy = extract_hdf5_metrics(str(greedy_out), '')

        out = {
            "seed": seed,
            "preproc": {"cmd": cmd_pre, "result": res_pre, "hdf5": metrics_pre},
            "greedy": {"cmd": cmd_greedy, "result": res_greedy, "hdf5": metrics_greedy},
        }

        out_path = RESULT_DIR / f"compare_C{args.C}_seed{seed}.json"
        with open(out_path, 'w') as fh:
            json.dump(out, fh, indent=2)

        print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
