from datetime import datetime
import sys
import os
import time
import argparse

import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor, as_completed

from optimization.experiments_settings import (
    load_tanner_graph,
    parse_edgelist,
    codes,
    path_to_initial_codes,
    textfiles,
    noise_levels,
)

from optimization.logical_guided_search.logical_guided_eval import (
    get_code_parameters_and_matrices,
    evaluate_mc,
)
from optimization.logical_guided_search.logical_guided_hdf5 import (
    append_to_hdf5,
    update_hdf5_row,
)
from optimization.logical_guided_search.logical_guided_search_core import (
    improve_state_by_breaking_low_weight_logical,
)

OUTPUT_FILE = "optimization/results/logical_guided_search.hdf5"
SEARCH_STEPS = 20
TRIALS_PER_STEP = 100
BUDGET_SCREENING = 10_000
BUDGET_PRECISION = 100_000
TOP_K_PRECISION = 10


def state_key(state):
    return tuple(parse_edgelist(state).flatten().tolist())

from optimization.logical_guided_search.logical_guided_eval import (
    get_code_parameters_and_matrices,
    evaluate_mc,
)

def evaluate_candidate_task(task):
    state = task["state"]
    p = task["p"]
    budget = task["budget"]
    run_label = task["run_label"]

    params, Hx, Hz = get_code_parameters_and_matrices(state)
    ler, std, runtime = evaluate_mc(Hx, Hz, p, budget, run_label=run_label)

    return {
        "row_idx": task["row_idx"],
        "ler": ler,
        "std": std,
        "runtime": runtime,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", default=0, type=int, help="Code family index")
    parser.add_argument("-p", default=None, type=float, help="Physical error rate")
    parser.add_argument("-S", default=SEARCH_STEPS, type=int, help="Search steps")
    parser.add_argument("-T", default=TRIALS_PER_STEP, type=int, help="Trials per step")
    parser.add_argument("--screen_budget", default=BUDGET_SCREENING, type=int)
    parser.add_argument("--prec_budget", default=BUDGET_PRECISION, type=int)
    parser.add_argument("--topk", default=TOP_K_PRECISION, type=int)
    parser.add_argument("--workers", default=1, type=int, help="Number of parallel worker processes")
    args = parser.parse_args()

    C = args.C
    p = noise_levels[C] if args.p is None else args.p

    print("\n--- LOGICAL GUIDED SEARCH ---")
    print(f"Code family: {codes[C]}")
    print(f"Noise level p = {p}")
    print(f"Search steps = {args.S}")
    print(f"Trials/step = {args.T}")
    print(f"Budgets: screening={args.screen_budget}, precision={args.prec_budget}")
    print(f"Workers = {args.workers}")

    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])
    start_time = time.time()

    with h5py.File(OUTPUT_FILE, "a") as f:
        grp = f.require_group(codes[C])
        run_name = (
            f"logical_guided_S{args.S}_T{args.T}_p{p}_"
            f"{args.screen_budget}screen_{args.prec_budget}prec_top{args.topk}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_grp = grp.require_group(run_name)

        run_grp.attrs.update({
            "search_steps": args.S,
            "trials_per_step": args.T,
            "p": p,
            "screen_budget": args.screen_budget,
            "prec_budget": args.prec_budget,
            "topk": args.topk,
        })

        current_state = initial_state
        params_init, Hx_init, Hz_init = get_code_parameters_and_matrices(current_state)
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
            distance_before=min(params_init["d_classical"], params_init["d_T_classical"]),
            distance_after=min(params_init["d_classical"], params_init["d_T_classical"]),
            edges_to_add=None,
            edges_to_remove=None,
        )

        all_candidates = [{
            "state": current_state,
            "params": params_init,
            "row_idx": idx_init,
            "dist": min(params_init["d_classical"], params_init["d_T_classical"]),
            "logical_weight": 0.0,
        }]

        global_max_dist = all_candidates[0]["dist"]

        print("\n>>> PHASE 1: Logical-guided search")
        for step in range(1, args.S + 1):
            print(f"\nStep {step}/{args.S} (global max dist so far = {global_max_dist})")

            result = improve_state_by_breaking_low_weight_logical(
                state=current_state,
                get_code_parameters_and_matrices=get_code_parameters_and_matrices,
                max_trials=args.T,
                require_distance_non_decrease=True,
                verbose=True,
            )

            state_to_save = result["state"]
            params = result["params"]
            d_cand = min(params["d_classical"], params["d_T_classical"])
            logical_weight = result["logical_weight"]

            edges = parse_edgelist(state_to_save).astype(np.uint32)
            row_idx = append_to_hdf5(
                run_grp,
                edge_list=edges,
                params=params,
                logical_weight=logical_weight,
                accepted=result["accepted"],
                step=step,
                trial=result["trial"],
                parent_idx=all_candidates[-1]["row_idx"] if step > 0 else idx_init,
                distance_before=result["distance_before"],
                distance_after=result["distance_after"],
                edges_to_add=result["edges_to_add"],
                edges_to_remove=result["edges_to_remove"],
            )

            all_candidates.append({
                "state": state_to_save,
                "params": params,
                "row_idx": row_idx,
                "dist": d_cand,
                "logical_weight": logical_weight,
            })

            if d_cand > global_max_dist:
                global_max_dist = d_cand

            if result["accepted"]:
                current_state = state_to_save

            f.flush()

        print(f"\n>>> PHASE 2: Screening ({args.screen_budget})")
        target_dist = global_max_dist
        screening_candidates = [c for c in all_candidates if c["dist"] == target_dist]

        unique_candidates_dict = {}
        for c in screening_candidates:
            key = state_key(c["state"])
            if key not in unique_candidates_dict:
                unique_candidates_dict[key] = c
        unique_candidates = list(unique_candidates_dict.values())

        print(f"Found {len(all_candidates)} total saved states.")
        print(f"Screening {len(unique_candidates)} unique candidates with dist == {target_dist}")

        print(f"\n>>> PHASE 2: Screening ({args.screen_budget})")
        target_dist = global_max_dist
        screening_candidates = [c for c in all_candidates if c["dist"] == target_dist]

        unique_candidates_dict = {}
        for c in screening_candidates:
            key = state_key(c["state"])
            if key not in unique_candidates_dict:
                unique_candidates_dict[key] = c
        unique_candidates = list(unique_candidates_dict.values())

        print(f"Found {len(all_candidates)} total saved states.")
        print(f"Screening {len(unique_candidates)} unique candidates with dist == {target_dist}")

        screened_results = []

        if args.workers == 1:
            for cand in tqdm(unique_candidates, desc="Running MC (screening)"):
                _, Hx, Hz = get_code_parameters_and_matrices(cand["state"])
                ler, std, run_t = evaluate_mc(
                    Hx, Hz, p, args.screen_budget,
                    run_label=f"screen_row_{cand['row_idx']}",
                )

                update_hdf5_row(run_grp, cand["row_idx"], ler, std, run_t)

                cand["ler"] = ler
                cand["std"] = std
                cand["runtime"] = run_t
                screened_results.append(cand)
                f.flush()
        else:
            tasks = [
                {
                    "state": cand["state"],
                    "row_idx": cand["row_idx"],
                    "p": p,
                    "budget": args.screen_budget,
                    "run_label": f"screen_row_{cand['row_idx']}",
                }
                for cand in unique_candidates
            ]

            results_by_row = {}

            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(evaluate_candidate_task, task) for task in tasks]

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Running MC (screening)"):
                    res = fut.result()
                    results_by_row[res["row_idx"]] = res

            for cand in unique_candidates:
                res = results_by_row[cand["row_idx"]]
                update_hdf5_row(run_grp, cand["row_idx"], res["ler"], res["std"], res["runtime"])

                cand["ler"] = res["ler"]
                cand["std"] = res["std"]
                cand["runtime"] = res["runtime"]
                screened_results.append(cand)
                f.flush()

        if not screened_results:
            print("No candidates to screen.")
            return

        screened_results.sort(key=lambda x: x["ler"])
        print(f"Best screening LER: {screened_results[0]['ler']:.6f} (dist={screened_results[0]['dist']})")

        print(f"\n>>> PHASE 3: Precision ({args.prec_budget})")
        top_candidates = screened_results[:args.topk]
        print(f"Promoting top {len(top_candidates)} candidates")

        final_best_cand = None
        min_final_ler = np.inf
        final_best_std = np.inf

        print(f"\n>>> PHASE 3: Precision ({args.prec_budget})")
        top_candidates = screened_results[:args.topk]
        print(f"Promoting top {len(top_candidates)} candidates")

        final_best_cand = None
        min_final_ler = np.inf
        final_best_std = np.inf

        if args.workers == 1:
            for cand in top_candidates:
                print(f"Evaluating row {cand['row_idx']} | dist={cand['dist']} | screening LER={cand['ler']:.6f}")
                _, Hx, Hz = get_code_parameters_and_matrices(cand["state"])

                ler, std, run_t = evaluate_mc(
                    Hx, Hz, p, args.prec_budget,
                    run_label=f"precision_row_{cand['row_idx']}",
                )

                update_hdf5_row(run_grp, cand["row_idx"], ler, std, run_t)
                print(f"  -> LER: {ler:.6f} ± {std:.6f} | runtime={run_t//3600}h {run_t%3600//60}m {run_t%60:.2f}s")

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
                }
                for cand in top_candidates
            ]

            results_by_row = {}

            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(evaluate_candidate_task, task) for task in tasks]

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Running MC (precision)"):
                    res = fut.result()
                    results_by_row[res["row_idx"]] = res

            for cand in top_candidates:
                res = results_by_row[cand["row_idx"]]
                update_hdf5_row(run_grp, cand["row_idx"], res["ler"], res["std"], res["runtime"])

                print(
                    f"Evaluating row {cand['row_idx']} | dist={cand['dist']} | "
                    f"screening LER={cand['ler']:.6f}"
                )
                print(f"  -> LER: {res['ler']:.6f} ± {res['std']:.6f} | runtime={res['runtime']:.2f}s")

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

            best_edges = parse_edgelist(final_best_cand["state"]).astype(np.uint32)

            if "best_state" in run_grp:
                del run_grp["best_state"]
            run_grp.create_dataset(
                "best_state",
                data=best_edges[np.newaxis, :],
                dtype=np.uint32,
            )
            run_grp.attrs["min_cost"] = min_final_ler
            run_grp.attrs["best_dist"] = final_best_cand["dist"]

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 3600:.2f} hours")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()