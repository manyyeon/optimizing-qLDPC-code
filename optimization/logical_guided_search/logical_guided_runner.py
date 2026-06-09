from optimization.logical_guided_search.logical_guided_search_core import (
    generate_logical_guided_candidates,
    format_score_info,
    beam_rank_key
)
from optimization.logical_guided_search.logical_guided_hdf5 import (
    append_to_hdf5,
    update_hdf5_row,
)
from optimization.logical_guided_search.logical_guided_eval import (
    get_code_parameters_and_matrices,
    evaluate_mc,
    compute_weighted_low_weight_score,
)
from optimization.experiments_settings import (
    load_tanner_graph,
    parse_edgelist,
    from_edgelist,
    codes,
    path_to_initial_codes,
    textfiles,
    noise_levels,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import sys
import os
import time
import argparse

import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SEARCH_STEPS = 50
TRIALS_PER_STEP = 100
BUDGET_SCREENING = 10_000
BUDGET_PRECISION = 100_000
TOP_K_PRECISION = 10
STOP_IF_NO_IMPROVEMENT_FOR = 100


def state_key(state):
    return tuple(parse_edgelist(state).flatten().tolist())


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

    # Many of your best_state datasets are saved as shape (1, E).
    # Convert that to shape (E,) before from_edgelist.
    if edge_list.ndim == 2 and edge_list.shape[0] == 1:
        edge_list = edge_list[0]

    return from_edgelist(edge_list)


def select_beam_with_backups(candidates, beam_width):
    """
    Distance-priority beam selection, but keep lower-distance backup states
    if there are fewer than beam_width states at the current max distance.
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=beam_rank_key)
    max_dist = max(int(c["dist"]) for c in candidates)

    selected = []
    selected_keys = set()

    # First take max-distance candidates.
    max_dist_candidates = [c for c in candidates if int(c["dist"]) == max_dist]

    for cand in max_dist_candidates:
        key = state_key(cand["state"])
        if key in selected_keys:
            continue
        selected.append(cand)
        selected_keys.add(key)
        if len(selected) >= beam_width:
            return selected

    # Then fill remaining slots with lower-distance backup states.
    for cand in candidates:
        key = state_key(cand["state"])
        if key in selected_keys:
            continue
        selected.append(cand)
        selected_keys.add(key)
        if len(selected) >= beam_width:
            break

    return selected


def score_top_count(n: int, frac: float, min_top: int, max_top: int) -> int:
    if n <= 0:
        return 0
    return min(max(int(np.ceil(frac * n)), min_top), max_top, n)


def evaluate_candidate_task(task):
    np.random.seed()

    state = task["state"]
    p = task["p"]
    budget = task["budget"]
    run_label = task["run_label"]
    failure_cap = task.get("failure_cap", None)
    min_runs_before_stop = task.get("min_runs_before_stop", 0)
    workers = task.get("workers", 1)
    batch_size = task.get("batch_size", 10000)

    _, Hx, Hz = get_code_parameters_and_matrices(state)

    result = evaluate_mc(
        Hx,
        Hz,
        p,
        budget,
        run_label=run_label,
        failure_cap=failure_cap,
        min_runs_before_stop=min_runs_before_stop,
        workers=workers,
        batch_size=batch_size,
    )

    return {
        "row_idx": task["row_idx"],
        "ler": result["ler"],
        "std": result["stderr"],
        "runtime": result["runtime"],
        "failures": result["failures"],
        "completed_runs": result["completed_runs"],
        "early_stopped": result["early_stopped"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", default=0, type=int, help="Code family index")
    parser.add_argument("-p", default=None, type=float,
                        help="Physical error rate")
    parser.add_argument("-S", default=SEARCH_STEPS,
                        type=int, help="Search steps")
    parser.add_argument("-T", default=TRIALS_PER_STEP,
                        type=int, help="Trials per step")
    parser.add_argument("--OUTPUT_FILE", default=None,
                        type=str, help="HDF5 file to save results")
    parser.add_argument("--screen_budget", default=BUDGET_SCREENING, type=int)
    parser.add_argument("--prec_budget", default=BUDGET_PRECISION, type=int)
    parser.add_argument("--topk", default=TOP_K_PRECISION, type=int)
    parser.add_argument("--workers", default=1, type=int,
                        help="Number of parallel worker processes")
    parser.add_argument("--score-beta", default=0.3, type=float)
    parser.add_argument("--score-window", default=2, type=int)
    parser.add_argument("--score-top-frac", default=0.10, type=float)
    parser.add_argument("--score-min-top", default=3, type=int)
    parser.add_argument("--score-max-top", default=5, type=int)
    parser.add_argument("--score-mode", default="absolute",
                        choices=["relative", "absolute"])

    parser.add_argument("--score-gamma", default=0.3, type=float)

    parser.add_argument("--score-max-weight", default=14, type=int)

    parser.add_argument("--rank-mode", default="score_only",
                        choices=["distance_first", "score_only"])
    parser.add_argument("--beam-width", default=5, type=int)
    parser.add_argument("--children-per-parent", default=20, type=int)
    parser.add_argument("--stop-distance", default=None, type=int)
    parser.add_argument("--candidate-workers", default=1, type=int)
    parser.add_argument(
        "--input-file",
        default=None,
        type=str,
        help="Optional HDF5 file to load the initial state from.",
    )
    parser.add_argument(
        "--input-code",
        default=None,
        type=str,
        help="Code group name in the input HDF5 file, e.g. '[625,25]'.",
    )
    parser.add_argument(
        "--input-run-name",
        default=None,
        type=str,
        help="Optional run group name under the code group.",
    )
    parser.add_argument(
        "--input-dataset",
        default="best_state",
        type=str,
        help="Dataset name containing the initial state edge list.",
    )

    args = parser.parse_args()

    C = args.C
    p = noise_levels[C] if args.p is None else args.p
    OUTPUT_FILE = (
        args.OUTPUT_FILE
        if args.OUTPUT_FILE is not None
        else "optimization/results/logical_guided_search.hdf5"
    )

    def score_candidate(state, params=None):
        return compute_weighted_low_weight_score(
            state=state,
            params=params,
            beta=args.score_beta,
            max_weight_offset=args.score_window,
            score_mode=args.score_mode,
            gamma=args.score_gamma,
            max_weight=args.score_max_weight,
        )

    print("\n--- LOGICAL GUIDED SEARCH ---")
    print(f"Code family: {codes[C]}")
    print(f"Noise level p = {p}")
    print(f"Search steps = {args.S}")
    print(f"Trials/step = {args.T}")
    print(f"Beam width = {args.beam_width}")
    print(f"Precision budget = {args.prec_budget}")
    print(
        f"Score selection: mode={args.score_mode}, "
        f"beta={args.score_beta}, "
        f"window={args.score_window}, "
        f"gamma={args.score_gamma}, "
        f"max_weight={args.score_max_weight}, "
        f"rank_mode={args.rank_mode}, "
        f"top_frac={args.score_top_frac}, "
        f"min_top={args.score_min_top}, "
        f"max_top={args.score_max_top}"
    )
    print(f"Workers = {args.workers}")
    print(f"Output HDF5: {OUTPUT_FILE}")
    print(f"Stop if no improvement for {STOP_IF_NO_IMPROVEMENT_FOR} steps")

    if args.input_file is not None:
        input_code = args.input_code if args.input_code is not None else codes[C]

        print(f"Loading initial state from HDF5:")
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
        initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    start_time = time.time()

    with h5py.File(OUTPUT_FILE, "a") as f:
        grp = f.require_group(codes[C])
        run_name = (
            f"logical_guided_score_S{args.S}_T{args.T}_p{p}_bw{args.beam_width}_"
            f"beta{args.score_beta}_win{args.score_window}_"
            f"scoretop{args.score_top_frac}_min{args.score_min_top}_max{args.score_max_top}_"
            f"{args.prec_budget}prec_"
            f"gamma{args.score_gamma}_maxw{args.score_max_weight}_"
            f"rank{args.rank_mode}_beam{args.beam_width}_children{args.children_per_parent}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_grp = grp.require_group(run_name)

        run_grp.attrs.update({
            "search_steps": args.S,
            "trials_per_step": args.T,
            "p": p,
            "screen_budget": args.screen_budget,
            "prec_budget": args.prec_budget,
            "beam_width": args.beam_width,
            "children_per_parent": args.children_per_parent,
            "topk": args.topk,
            "score_mode": args.score_mode,
            "score_beta": args.score_beta,
            "score_window": args.score_window,
            "score_gamma": args.score_gamma,
            "score_max_weight": args.score_max_weight,
            "rank_mode": args.rank_mode,
            "score_top_frac": args.score_top_frac,
            "score_min_top": args.score_min_top,
            "score_max_top": args.score_max_top,
        })

        current_state = initial_state
        params_init, Hx_init, Hz_init = get_code_parameters_and_matrices(
            current_state)
        score_info_init = score_candidate(current_state, params_init)
        edges_init = parse_edgelist(current_state).astype(np.uint32)

        idx_init = append_to_hdf5(
            run_grp,
            edge_list=edges_init,
            params=params_init,
            logical_weight=0.0,
            accepted=True,
            step=0,
            trial=0,
            parent_idx=-1,
            distance_before=min(
                params_init["d_classical"], params_init["d_T_classical"]),
            distance_after=min(
                params_init["d_classical"], params_init["d_T_classical"]),
            edges_to_add=None,
            edges_to_remove=None,
        )

        current_parent_row_idx = idx_init

        all_candidates = [{
            "state": current_state,
            "params": params_init,
            "row_idx": idx_init,
            "dist": min(params_init["d_classical"], params_init["d_T_classical"]),
            "logical_weight": 0.0,
            "low_weight_score": score_info_init["score"],
            "score_info": score_info_init,
        }]

        last_improvement_step = 0
        global_max_dist = all_candidates[0]["dist"]

        print(f"\n>>> PHASE 1: Beam {args.beam_width} Logical-guided search")

        beam = [all_candidates[0]]
        seen_state_keys = {state_key(current_state)}

        for step in range(1, args.S + 1):
            print(
                f"\nBeam depth {step}/{args.S} (global max dist so far = {global_max_dist})")

            print("Current beam:")
            for b_i, cand in enumerate(beam):
                print(
                    f"  beam[{b_i}] row={cand['row_idx']} | "
                    f"dist={cand['dist']} | "
                    f"{format_score_info(cand.get('score_info'))}"
                )

            children = []

            for parent_i, parent in enumerate(beam):
                print(
                    f"\nExpanding parent beam[{parent_i}] row={parent['row_idx']}")

                raw_children = generate_logical_guided_candidates(
                    state=parent["state"],
                    get_code_parameters_and_matrices=get_code_parameters_and_matrices,
                    max_trials=args.T,
                    logical_max_comb_order=5,
                    require_detectable=True,
                    require_distance_non_decrease=True,
                    verbose=True,
                    seen_keys=seen_state_keys,
                    score_candidate_fn=score_candidate,
                )

                appended_count = 0

                for child_i, result in enumerate(raw_children):
                    state_to_save = result["state"]
                    key = state_key(state_to_save)

                    if key in seen_state_keys:
                        continue

                    seen_state_keys.add(key)

                    params = result["params"]
                    d_cand = min(params["d_classical"],
                                 params["d_T_classical"])
                    logical_weight = result["logical_weight"]

                    score_info = result.get("score_info")
                    if score_info is None:
                        score_info = score_candidate(state_to_save, params)

                    low_weight_score = float(score_info["score"])

                    edges = parse_edgelist(state_to_save).astype(np.uint32)
                    row_idx = append_to_hdf5(
                        run_grp,
                        edge_list=edges,
                        params=params,
                        logical_weight=logical_weight,
                        accepted=True,
                        step=step,
                        trial=result["trial"],
                        parent_idx=parent["row_idx"],
                        distance_before=result["distance_before"],
                        distance_after=result["distance_after"],
                        edges_to_add=result["edges_to_add"],
                        edges_to_remove=result["edges_to_remove"],
                    )

                    cand = {
                        "state": state_to_save,
                        "params": params,
                        "row_idx": row_idx,
                        "parent_idx": parent["row_idx"],
                        "dist": d_cand,
                        "logical_weight": logical_weight,
                        "low_weight_score": low_weight_score,
                        "score_info": score_info,
                    }

                    children.append(cand)
                    all_candidates.append(cand)
                    appended_count += 1

                    if d_cand > global_max_dist:
                        global_max_dist = d_cand
                        last_improvement_step = step

                    print(
                        f"  child row={row_idx} | "
                        f"parent={parent['row_idx']} | "
                        f"dist={d_cand} | "
                        f"target_weight_before_swap={logical_weight} | "
                        f"{format_score_info(score_info)}"
                    )

                    f.flush()

                print(
                    f"  parent row={parent['row_idx']} appended "
                    f"{appended_count}/{len(raw_children)} children"
                )

            if not children:
                print("No new children generated.")
                continue

            # Deduplicate children again just in case.
            unique_pool = {}
            for cand in children:
                key = state_key(cand["state"])
                if key not in unique_pool:
                    unique_pool[key] = cand
                elif beam_rank_key(cand) < beam_rank_key(unique_pool[key]):
                    unique_pool[key] = cand

            candidate_pool = list(unique_pool.values())
            candidate_pool.sort(key=beam_rank_key)

            beam = select_beam_with_backups(candidate_pool, args.beam_width)

            print("\nSelected next beam:")
            for b_i, cand in enumerate(beam):
                print(
                    f"  beam[{b_i}] row={cand['row_idx']} | "
                    f"dist={cand['dist']} | "
                    f"parent={cand.get('parent_idx', -1)} | "
                    f"{format_score_info(cand.get('score_info'))}"
                )

            if args.stop_distance is not None and global_max_dist >= args.stop_distance:
                print(
                    f"Reached stop distance {args.stop_distance}. Stopping beam search.")
                break

            if step - last_improvement_step >= STOP_IF_NO_IMPROVEMENT_FOR:
                print(
                    f"Distance has not increased in the last "
                    f"{STOP_IF_NO_IMPROVEMENT_FOR} steps, stopping early."
                )
                break

        print(f"\n>>> PHASE 2: Weighted-score selection")
        if args.rank_mode == "distance_first":
            target_dist = global_max_dist
            score_candidates = [
                c for c in all_candidates if c["dist"] == target_dist]
            print(f"Selecting among candidates with dist == {target_dist}.")

        else:
            target_dist = None
            score_candidates = list(all_candidates)
            print("Selecting among all candidates by absolute score only.")

        unique_candidates_dict = {}

        for c in score_candidates:
            key = state_key(c["state"])
            if key not in unique_candidates_dict:
                unique_candidates_dict[key] = c

        unique_candidates = list(unique_candidates_dict.values())

        for cand in unique_candidates:
            if "low_weight_score" not in cand or not np.isfinite(cand["low_weight_score"]):
                score_info = score_candidate(cand["state"], cand["params"])
                cand["score_info"] = score_info
                cand["low_weight_score"] = float(score_info["score"])

        unique_candidates.sort(
            key=lambda c: (
                float(c.get("low_weight_score", np.inf)),
                -int(c.get("dist", -1)),
                float(c.get("logical_weight", np.inf)),
                int(c.get("row_idx", 10**18)),
            )
        )

        num_to_eval = score_top_count(
            len(unique_candidates),
            frac=args.score_top_frac,
            min_top=args.score_min_top,
            max_top=args.score_max_top,
        )

        top_candidates = unique_candidates[:num_to_eval]

        print(f"Found {len(all_candidates)} total saved states.")

        print(
            f"Found {len(unique_candidates)} unique candidates for score selection.")
        print(
            f"Selecting top {num_to_eval} candidates by "
            f"{args.score_mode} weighted score "
            f"(gamma={args.score_gamma}, max_weight={args.score_max_weight})."
        )

        for cand in top_candidates:
            print(
                f"  row={cand['row_idx']} | "
                f"dist={cand['dist']} | "
                f"target_weight_before_swap={cand['logical_weight']} | "
                f"{format_score_info(cand.get('score_info'))}"
            )

        if not top_candidates:
            print("No candidates selected for precision evaluation.")
            return

        print(f"\n>>> PHASE 3: Precision LER evaluation ({args.prec_budget})")
        final_best_cand = None
        min_final_ler = np.inf
        final_best_std = np.inf

        if args.workers == 1:
            for cand in top_candidates:
                print(
                    f"Evaluating row {cand['row_idx']} | "
                    f"dist={cand['dist']} | "
                    f"target_weight={cand['logical_weight']} | "
                    f"{format_score_info(cand.get('score_info'))}"
                )

                _, Hx, Hz = get_code_parameters_and_matrices(cand["state"])

                result = evaluate_mc(
                    Hx,
                    Hz,
                    p,
                    args.prec_budget,
                    run_label=f"precision_row_{cand['row_idx']}",
                )

                ler = result["ler"]
                std = result["stderr"]
                run_t = result["runtime"]

                update_hdf5_row(run_grp, cand["row_idx"], ler, std, run_t)

                print(
                    f"  -> LER: {ler:.6f} ± {std:.6f} | "
                    f"runtime={run_t//3600}h {run_t % 3600//60}m {run_t % 60:.2f}s"
                )

                cand["prec_ler"] = ler
                cand["prec_std"] = std
                cand["prec_runtime"] = run_t

                if ler < min_final_ler:
                    min_final_ler = ler
                    final_best_std = std
                    final_best_cand = cand

                f.flush()

        else:
            tasks = [
                {
                    "state": cand["state"],
                    "row_idx": cand["row_idx"],
                    "p": p,
                    "budget": args.prec_budget,
                    "run_label": f"precision_row_{cand['row_idx']}",
                    "workers": args.workers,
                    "batch_size": 10000,
                }
                for cand in top_candidates
            ]

            results_by_row = {}

            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(evaluate_candidate_task, task)
                           for task in tasks]

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Running MC (precision)",
                ):
                    res = fut.result()
                    results_by_row[res["row_idx"]] = res

            for cand in top_candidates:
                res = results_by_row[cand["row_idx"]]

                update_hdf5_row(
                    run_grp,
                    cand["row_idx"],
                    res["ler"],
                    res["std"],
                    res["runtime"],
                )

                print(
                    f"Evaluating row {cand['row_idx']} | "
                    f"dist={cand['dist']} | "
                    f"score={cand['low_weight_score']:.6g} | "
                    f"{format_score_info(cand.get('score_info'))}"
                )
                print(
                    f"  -> LER: {res['ler']:.6f} ± {res['std']:.6f} | "
                    f"runtime={res['runtime']:.2f}s"
                )

                cand["prec_ler"] = res["ler"]
                cand["prec_std"] = res["std"]
                cand["prec_runtime"] = res["runtime"]

                if res["ler"] < min_final_ler:
                    min_final_ler = res["ler"]
                    final_best_std = res["std"]
                    final_best_cand = cand

                f.flush()

        if final_best_cand is not None:
            print("\n>>> RESULT: Best code found")
            print(f"Distance: {final_best_cand['dist']}")
            print(f"LER: {min_final_ler:.6f} ± {final_best_std:.6f}")

            best_edges = parse_edgelist(
                final_best_cand["state"]).astype(np.uint32)

            if "best_state" in run_grp:
                del run_grp["best_state"]
            run_grp.create_dataset(
                "best_state",
                data=best_edges[np.newaxis, :],
                dtype=np.uint32,
            )
            run_grp.attrs["min_cost"] = min_final_ler
            run_grp.attrs["best_dist"] = final_best_cand["dist"]

        print(f"Run group name: {run_grp.name}")

    total_time = time.time() - start_time
    print(
        f"\nTotal time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.2f}s")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
