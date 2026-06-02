from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from datetime import datetime

import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from optimization.experiments_settings import (
    load_tanner_graph,
    from_edgelist,
    codes,
    path_to_initial_codes,
    textfiles,
)
from optimization.experiments_settings import tanner_graph_to_parity_check_matrix

from optimization.logical_guided_search.logical_guided_rl_env import LogicalGuidedSwapEnv


def load_initial_state_from_hdf5(
    input_file: str,
    input_code: str,
    input_run_name: str | None = None,
    input_dataset: str = "best_state",
):
    with h5py.File(input_file, "r") as f:
        if input_code not in f:
            raise KeyError(
                f"{input_code!r} not found in {input_file}. "
                f"Available code groups: {list(f.keys())}"
            )

        grp = f[input_code]

        if input_run_name is not None:
            if input_run_name not in grp:
                raise KeyError(
                    f"{input_run_name!r} not found under {input_code!r}. "
                    f"Available groups: {list(grp.keys())}"
                )
            grp = grp[input_run_name]

        if input_dataset not in grp:
            raise KeyError(
                f"{input_dataset!r} not found in input group. "
                f"Available datasets/keys: {list(grp.keys())}"
            )

        edge_list = np.asarray(grp[input_dataset][:])

    if edge_list.ndim == 2 and edge_list.shape[0] == 1:
        edge_list = edge_list[0]

    return from_edgelist(edge_list)


def degree_summary(state):
    """
    Effective degree summary from the parity-check matrix H.
    This is important if your NetworkX graph is a MultiGraph and duplicate
    edges are allowed, because H may collapse parallel edges depending on
    tanner_graph_to_parity_check_matrix.
    """
    H = np.asarray(tanner_graph_to_parity_check_matrix(state), dtype=np.uint8)

    row_weights = H.sum(axis=1)
    col_weights = H.sum(axis=0)

    return {
        "max_check_weight": int(row_weights.max()) if row_weights.size else 0,
        "min_check_weight": int(row_weights.min()) if row_weights.size else 0,
        "avg_check_weight": float(row_weights.mean()) if row_weights.size else 0.0,
        "max_var_degree": int(col_weights.max()) if col_weights.size else 0,
        "min_var_degree": int(col_weights.min()) if col_weights.size else 0,
        "avg_var_degree": float(col_weights.mean()) if col_weights.size else 0.0,
        "num_edges_effective": int(H.sum()),
    }


def valid_actions_from_obs(obs):
    mask = np.asarray(obs["mask"], dtype=np.float32)
    return np.flatnonzero(mask > 0.5)


def run_random_episode(env, rng: np.random.Generator, episode: int, verbose: bool = True):
    obs = env.reset()
    history = []

    cumulative_reward = 0.0

    for step in range(env.max_steps):
        valid_actions = valid_actions_from_obs(obs)

        if len(valid_actions) == 0:
            if verbose:
                print(f"  step={step}: no valid candidates, stopping episode.")
            break

        action = env._choose_greedy_action()

        old_score = float(env.score_info["score"])
        old_params = dict(env.params)
        old_degree = degree_summary(env.state)
        old_weight_patterns = env.score_info.get("components", None)

        obs_next, reward, done, info = env.step(action)

        new_score = float(info["score"])
        new_degree = degree_summary(env.state)
        new_weight_patterns = info.get("weight_patterns", None)

        cumulative_reward += float(reward)

        row = {
            "episode": episode,
            "step": step,
            "action": action,
            "num_valid_actions": int(len(valid_actions)),
            "reward": float(reward),
            "cumulative_reward": float(cumulative_reward),

            "old_score": old_score,
            "new_score": new_score,
            "score_delta": new_score - old_score,
            "log_score_improvement": float(np.log1p(old_score) - np.log1p(new_score)),

            "old_d_quantum": int(old_params["d_quantum"]),
            "new_d_quantum": int(info["d_quantum"]),
            "old_k_quantum": int(old_params["k_quantum"]),
            "new_k_quantum": int(info["k_quantum"]),
            "old_rank_H": int(old_params["rank_H"]),
            "new_rank_H": int(info["rank_H"]),
            "old_weight_patterns": old_weight_patterns,
            "new_weight_patterns": new_weight_patterns,

            "old_max_check_weight": old_degree["max_check_weight"],
            "new_max_check_weight": new_degree["max_check_weight"],
            "old_max_var_degree": old_degree["max_var_degree"],
            "new_max_var_degree": new_degree["max_var_degree"],
            "old_num_edges_effective": old_degree["num_edges_effective"],
            "new_num_edges_effective": new_degree["num_edges_effective"],

            "logical_weight": int(info.get("logical_weight", -1)),
            "edges_to_add": repr(info.get("edges_to_add", None)),
            "edges_to_remove": repr(info.get("edges_to_remove", None)),
        }

        history.append(row)

        if verbose:
            print(
                f"  step={step:03d} | "
                f"valid={len(valid_actions):03d} | "
                f"action={action:02d} | "
                f"reward={reward:+.4f} | "
                f"score {old_score:.6g}->{new_score:.6g} | "
                f"weight patterns {row['old_weight_patterns']}->{row['new_weight_patterns']} | "
                f"d {row['old_d_quantum']}->{row['new_d_quantum']} | "
                f"k {row['old_k_quantum']}->{row['new_k_quantum']} | "
                f"rank {row['old_rank_H']}->{row['new_rank_H']} | "
                f"wmax {row['old_max_check_weight']}->{row['new_max_check_weight']} | "
                f"qmax {row['old_max_var_degree']}->{row['new_max_var_degree']}"
            )

        if done:
            break

        obs = obs_next

    return history


def write_csv(rows, output_csv):
    if not rows:
        print("No rows to save.")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved random-policy history to {output_csv}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-C", default=0, type=int, help="Code family index")
    parser.add_argument("--episodes", default=3, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--candidates", default=32, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--score-beta", default=0.3, type=float)
    parser.add_argument("--score-window", default=2, type=int)

    parser.add_argument("--require-k-preserved", default=True, action="store_true")
    parser.add_argument("--quiet", default=False, action="store_true")

    parser.add_argument("--input-file", default=None, type=str)
    parser.add_argument("--input-code", default=None, type=str)
    parser.add_argument("--input-run-name", default=None, type=str)
    parser.add_argument("--input-dataset", default="best_state", type=str)

    parser.add_argument(
        "--output-csv",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    C = args.C
    input_code = args.input_code if args.input_code is not None else codes[C]

    if args.output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = (
            f"optimization/results/rl_random_sanity_"
            f"C{C}_S{args.steps}_K{args.candidates}_seed{args.seed}_{timestamp}.csv"
        )

    if args.input_file is not None:
        print("Loading initial state from HDF5:")
        print(f"  input_file={args.input_file}")
        print(f"  input_code={input_code}")
        print(f"  input_run_name={args.input_run_name}")
        print(f"  input_dataset={args.input_dataset}")

        initial_state = load_initial_state_from_hdf5(
            input_file=args.input_file,
            input_code=input_code,
            input_run_name=args.input_run_name,
            input_dataset=args.input_dataset,
        )
    else:
        print("Loading initial state from text file:")
        print(f"  code={codes[C]}")
        print(f"  file={path_to_initial_codes + textfiles[C]}")
        initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    print("\n--- RANDOM VALID-CANDIDATE RL SANITY CHECK ---")
    print(f"Code family: {codes[C]}")
    print(f"Episodes: {args.episodes}")
    print(f"Steps per episode: {args.steps}")
    print(f"Candidate slots K: {args.candidates}")
    print(f"Score beta: {args.score_beta}")
    print(f"Score window: {args.score_window}")
    print(f"Require k preserved: {args.require_k_preserved}")
    print(f"Seed: {args.seed}")
    print(f"Output CSV: {args.output_csv}")

    all_rows = []
    start = time.time()

    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}")

        random.seed(args.seed + ep)
        rng = np.random.default_rng(args.seed + ep)

        env = LogicalGuidedSwapEnv(
            initial_state=initial_state,
            max_steps=args.steps,
            candidates_per_step=args.candidates,
            score_beta=args.score_beta,
            score_window=args.score_window,
            require_k_preserved=args.require_k_preserved,
        )

        rows = run_random_episode(
            env=env,
            rng=rng,
            episode=ep,
            verbose=not args.quiet,
        )

        if rows:
            print(
                f"Episode {ep + 1} finished: "
                f"steps={len(rows)}, "
                f"final_score={rows[-1]['new_score']:.6g}, "
                f"cum_reward={rows[-1]['cumulative_reward']:+.4f}, "
                f"final_d={rows[-1]['new_d_quantum']}, "
                f"final_k={rows[-1]['new_k_quantum']}"
            )
        else:
            print(f"Episode {ep + 1} produced no valid steps.")

        all_rows.extend(rows)

    write_csv(all_rows, args.output_csv)

    elapsed = time.time() - start
    print(f"\nDone. Runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()