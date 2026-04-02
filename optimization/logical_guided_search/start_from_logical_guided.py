from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import sys
import os
import time
import argparse

import h5py
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.experiments_settings import (
    codes,
    noise_levels,
    parse_edgelist,
    from_edgelist,
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
    generate_logical_guided_candidates,
)

OUTPUT_FILE = "optimization/results/start_from_logical_guided.hdf5"
BEAM_WIDTH = 3
DEPTH = 10
EXPAND_PER_PARENT = 12
TOPK_PER_PARENT_FOR_MC = 5
SCREEN_BUDGET = 10_000
PREC_BUDGET = 100_000


def state_key(state):
    return tuple(parse_edgelist(state).flatten().tolist())


def load_best_state_from_logical_guided(input_file, code_name, run_name):
    with h5py.File(input_file, "r") as f:
        grp = f[code_name][run_name]
        best_state_edge_list = grp["best_state"][0]
    return from_edgelist(best_state_edge_list)


def evaluate_candidate_task(task):
    state = task["state"]
    p = task["p"]
    budget = task["budget"]
    run_label = task["run_label"]

    _, Hx, Hz = get_code_parameters_and_matrices(state)
    ler, std, rt = evaluate_mc(Hx, Hz, p, budget, run_label=run_label)

    return {
        "candidate_id": task["candidate_id"],
        "ler": ler,
        "std": std,
        "runtime": rt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", default=0, type=int)
    parser.add_argument("-p", default=None, type=float)
    parser.add_argument("--input_file", default="optimization/results/logical_guided_search.hdf5")
    parser.add_argument("--input_run", required=True, type=str)
    parser.add_argument("--beam_width", default=BEAM_WIDTH, type=int)
    parser.add_argument("--depth", default=DEPTH, type=int)
    parser.add_argument("--expand_per_parent", default=EXPAND_PER_PARENT, type=int)
    parser.add_argument("--topk_per_parent", default=TOPK_PER_PARENT_FOR_MC, type=int)
    parser.add_argument("--screen_budget", default=SCREEN_BUDGET, type=int)
    parser.add_argument("--prec_budget", default=PREC_BUDGET, type=int)
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()

    C = args.C
    code_name = codes[C]
    p = noise_levels[C] if args.p is None else args.p

    print("\n--- BEAM SEARCH FROM LOGICAL-GUIDED BEST STATE ---")
    print(f"Code family: {code_name}")
    print(f"Noise level p = {p}")
    print(f"Beam width = {args.beam_width}")
    print(f"Depth = {args.depth}")
    print(f"Expand per parent = {args.expand_per_parent}")
    print(f"Top-k per parent for MC = {args.topk_per_parent}")
    print(f"Budgets: screen={args.screen_budget}, precision={args.prec_budget}")
    print(f"Workers = {args.workers}")
    print(f"Source run = {args.input_run}")

    initial_state = load_best_state_from_logical_guided(
        args.input_file,
        code_name,
        args.input_run,
    )

    start_time = time.time()

    with h5py.File(OUTPUT_FILE, "a") as f:
        grp = f.require_group(code_name)
        run_name = (
            f"beam_from_logical_bw{args.beam_width}_d{args.depth}_"
            f"exp{args.expand_per_parent}_topk{args.topk_per_parent}_p{p}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_grp = grp.require_group(run_name)

        run_grp.attrs.update({
            "beam_width": args.beam_width,
            "depth": args.depth,
            "expand_per_parent": args.expand_per_parent,
            "topk_per_parent": args.topk_per_parent,
            "p": p,
            "screen_budget": args.screen_budget,
            "prec_budget": args.prec_budget,
            "workers": args.workers,
            "source_run": args.input_run,
        })

        params0, Hx0, Hz0 = get_code_parameters_and_matrices(initial_state)
        d0 = min(params0["d_classical"], params0["d_T_classical"])
        ler0, std0, rt0 = evaluate_mc(Hx0, Hz0, p, args.screen_budget, run_label="beam_init")

        idx0 = append_to_hdf5(
            run_grp,
            edge_list=parse_edgelist(initial_state).astype(np.uint32),
            params=params0,
            ler=ler0,
            std=std0,
            runtime=rt0,
            logical_weight=0.0,
            accepted=True,
            step=0,
            trial=0,
            parent_idx=-1,
            distance_before=d0,
            distance_after=d0,
            edges_to_add=None,
            edges_to_remove=None,
        )

        current_beam = [{
            "state": initial_state,
            "params": params0,
            "row_idx": idx0,
            "dist": d0,
            "ler": ler0,
            "std": std0,
        }]

        best_global = current_beam[0]

        print(f"\nInitial state: dist={d0}, LER={ler0:.6f}")

        for depth in range(1, args.depth + 1):
            print(f"\n=== Beam depth {depth}/{args.depth} ===")
            all_children = []
            seen = set()
            pending_candidates = []

            # Stage A: generate all unique children serially
            for parent_i, parent in enumerate(current_beam):
                print(
                    f"  Parent {parent_i + 1}/{len(current_beam)} | "
                    f"row={parent['row_idx']} | dist={parent['dist']} | ler={parent['ler']:.6f}"
                )

                raw_candidates = generate_logical_guided_candidates(
                    state=parent["state"],
                    get_code_parameters_and_matrices=get_code_parameters_and_matrices,
                    num_candidates=args.expand_per_parent,
                    proposal_max_tries=200,
                    logical_max_comb_order=5,
                    require_detectable=True,
                    require_distance_non_decrease=False,
                    verbose=False,
                )

                raw_candidates.sort(
                    key=lambda c: (c["distance_after"], -c["logical_weight"]),
                    reverse=True,
                )
                raw_candidates = raw_candidates[:args.topk_per_parent]

                for trial_idx, cand in enumerate(raw_candidates):
                    key = state_key(cand["state"])
                    if key in seen:
                        continue
                    seen.add(key)

                    candidate_id = len(pending_candidates)
                    pending_candidates.append({
                        "candidate_id": candidate_id,
                        "cand": cand,
                        "parent_row_idx": parent["row_idx"],
                        "trial_idx": trial_idx,
                        "run_label": f"beam_d{depth}_parent{parent['row_idx']}_trial{trial_idx}",
                    })

            if not pending_candidates:
                print("No children generated. Stopping.")
                break

            # Stage B: evaluate all children (serial or parallel)
            results_by_id = {}

            if args.workers == 1:
                for item in tqdm(pending_candidates, desc=f"Beam depth {depth} MC"):
                    cand = item["cand"]
                    _, Hx, Hz = get_code_parameters_and_matrices(cand["state"])
                    ler, std, rt = evaluate_mc(
                        Hx, Hz, p, args.screen_budget, run_label=item["run_label"]
                    )
                    results_by_id[item["candidate_id"]] = {
                        "ler": ler,
                        "std": std,
                        "runtime": rt,
                    }
            else:
                tasks = [
                    {
                        "candidate_id": item["candidate_id"],
                        "state": item["cand"]["state"],
                        "p": p,
                        "budget": args.screen_budget,
                        "run_label": item["run_label"],
                    }
                    for item in pending_candidates
                ]

                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futures = [ex.submit(evaluate_candidate_task, task) for task in tasks]

                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Beam depth {depth} MC",
                    ):
                        res = fut.result()
                        results_by_id[res["candidate_id"]] = {
                            "ler": res["ler"],
                            "std": res["std"],
                            "runtime": res["runtime"],
                        }

            # Stage C: save results in main process only
            for item in pending_candidates:
                cand = item["cand"]
                res = results_by_id[item["candidate_id"]]

                row_idx = append_to_hdf5(
                    run_grp,
                    edge_list=parse_edgelist(cand["state"]).astype(np.uint32),
                    params=cand["params"],
                    ler=res["ler"],
                    std=res["std"],
                    runtime=res["runtime"],
                    logical_weight=cand["logical_weight"],
                    accepted=True,
                    step=depth,
                    trial=item["trial_idx"],
                    parent_idx=item["parent_row_idx"],
                    distance_before=cand["distance_before"],
                    distance_after=cand["distance_after"],
                    edges_to_add=cand["edges_to_add"],
                    edges_to_remove=cand["edges_to_remove"],
                )

                child = {
                    "state": cand["state"],
                    "params": cand["params"],
                    "row_idx": row_idx,
                    "dist": cand["distance_after"],
                    "ler": res["ler"],
                    "std": res["std"],
                    "logical_weight": cand["logical_weight"],
                }
                all_children.append(child)

                if res["ler"] < best_global["ler"]:
                    best_global = child
                    print(f"    [!] New best global LER: {res['ler']:.6f}, dist={cand['distance_after']}")

                f.flush()

            # keep parents too, so search is non-regressive
            pool = list(all_children) + list(current_beam)
            pool.sort(key=lambda x: x["ler"])
            current_beam = pool[:args.beam_width]

            print("  Survivors:")
            for i, s in enumerate(current_beam):
                print(f"    {i+1}. row={s['row_idx']} dist={s['dist']} ler={s['ler']:.6f}")

        print("\n=== Precision evaluation on final beam ===")
        final_candidates = sorted(current_beam, key=lambda x: x["ler"])

        best_precise = None
        best_precise_ler = np.inf
        best_precise_std = np.inf

        if args.workers == 1:
            for cand in final_candidates:
                _, Hx, Hz = get_code_parameters_and_matrices(cand["state"])
                ler, std, rt = evaluate_mc(
                    Hx, Hz, p, args.prec_budget,
                    run_label=f"precision_row_{cand['row_idx']}",
                )
                update_hdf5_row(run_grp, cand["row_idx"], ler, std, rt)
                print(f"row={cand['row_idx']} dist={cand['dist']} precise LER={ler:.6f} ± {std:.6f}")

                if ler < best_precise_ler:
                    best_precise_ler = ler
                    best_precise_std = std
                    best_precise = cand

                f.flush()
        else:
            tasks = [
                {
                    "candidate_id": i,
                    "state": cand["state"],
                    "p": p,
                    "budget": args.prec_budget,
                    "run_label": f"precision_row_{cand['row_idx']}",
                }
                for i, cand in enumerate(final_candidates)
            ]

            results_by_id = {}

            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(evaluate_candidate_task, task) for task in tasks]

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Precision MC"):
                    res = fut.result()
                    results_by_id[res["candidate_id"]] = res

            for i, cand in enumerate(final_candidates):
                res = results_by_id[i]
                update_hdf5_row(run_grp, cand["row_idx"], res["ler"], res["std"], res["runtime"])
                print(
                    f"row={cand['row_idx']} dist={cand['dist']} "
                    f"precise LER={res['ler']:.6f} ± {res['std']:.6f}"
                )

                if res["ler"] < best_precise_ler:
                    best_precise_ler = res["ler"]
                    best_precise_std = res["std"]
                    best_precise = cand

                f.flush()

        if best_precise is not None:
            best_edges = parse_edgelist(best_precise["state"]).astype(np.uint32)
            if "best_state" in run_grp:
                del run_grp["best_state"]
            run_grp.create_dataset(
                "best_state",
                data=best_edges[np.newaxis, :],
                dtype=np.uint32,
            )
            run_grp.attrs["min_cost"] = best_precise_ler
            run_grp.attrs["best_dist"] = best_precise["dist"]

            print("\nBest final result:")
            print(f"  dist = {best_precise['dist']}")
            print(f"  LER  = {best_precise_ler:.6f} ± {best_precise_std:.6f}")

    total_time = time.time() - start_time
    print(f"\nTotal time: ({total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.2f}s)")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()