from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

import ldpc.code_util

from optimization.experiments_settings import (
    add_and_remove_edges,
    tanner_graph_to_parity_check_matrix,
)
from basic_css_code import construct_HGP_code
from logical_operators import get_logical_operators_by_pivoting

import sys
import os
import time
import argparse
import h5py
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ldpc
import ldpc.code_util
from ldpc.bposd_decoder import BpOsdDecoder

from decoder_performance import compute_logical_error_rate

from optimization.experiments_settings import (
    load_tanner_graph,
    parse_edgelist,
    tanner_graph_to_parity_check_matrix,
    codes,
    path_to_initial_codes,
    textfiles,
    noise_levels,
)

from optimization.experiments_settings import add_and_remove_edges

def compute_classical_distance_from_state(state: nx.MultiGraph) -> int:
    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)
    return ldpc.code_util.compute_exact_code_distance(csr_H)


def support_indices_to_graph_nodes(G: nx.MultiGraph, support: np.ndarray) -> set[int]:
    """
    support contains column indices of H.
    Convert them to actual variable-node labels in the Tanner graph.
    """
    variable_nodes = get_variable_nodes(G)
    return {variable_nodes[i] for i in support}


def run_logical_guided_search(
    initial_state: nx.MultiGraph,
    steps: int = 10,
    verbose: bool = True,
):
    state = nx.MultiGraph(initial_state)
    history = []

    for step in range(steps):
        if verbose:
            print(f"\n=== Step {step} ===")

        result = improve_state_by_breaking_low_weight_logical(
            state=state,
            max_trials=100,
            require_distance_non_decrease=True,
            verbose=verbose,
        )
        history.append(result)

        if not result["accepted"]:
            break

        state = result["state"]

    return state, history

# ============================================================
# CONFIG
# ============================================================

output_file = "optimization/results/logical_guided_search.hdf5"

# search parameters
SEARCH_STEPS = 20
TRIALS_PER_STEP = 100

# MC budgets
BUDGET_SCREENING = 10_000
BUDGET_PRECISION = 100_000
TOP_K_PRECISION = 10


# ============================================================
# PARAM / MC HELPERS
# ============================================================

def get_code_parameters_and_matrices(state: nx.MultiGraph):
    """Calculates parameters and returns matrices without running MC."""
    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)

    # Classical Parameters
    n_cl, k_cl, _ = ldpc.code_util.compute_code_parameters(csr_H)
    d_cl = ldpc.code_util.compute_exact_code_distance(csr_H)
    r_cl = ldpc.mod2.rank(csr_H)

    # Transpose Parameters
    if k_cl == n_cl - csr_H.shape[0]:
        n_T_cl = csr_H.shape[0]
        k_T_cl = n_T_cl - r_cl
        d_T_cl = np.inf
    else:
        n_T_cl, k_T_cl, d_T_cl = ldpc.code_util.compute_code_parameters(
            csr_matrix(H.T, dtype=np.uint8)
        )

    # Construct Quantum Code
    Hx, Hz = construct_HGP_code(H)

    n_q = n_cl**2 + n_T_cl**2
    k_q = k_cl**2 + k_T_cl**2
    d_q = min(d_cl, d_T_cl)

    params = {
        "n_classical": n_cl,
        "k_classical": k_cl,
        "d_classical": d_cl,
        "n_T_classical": n_T_cl,
        "k_T_classical": k_T_cl,
        "d_T_classical": d_T_cl,
        "rank_H": r_cl,
        "n_quantum": n_q,
        "k_quantum": k_q,
        "d_quantum": d_q,
    }
    return params, Hx, Hz


def evaluate_mc(Hx, Hz, p, budget, run_label="eval"):
    """Runs Monte Carlo simulation only."""
    if budget == 0:
        return 0.0, 0.0, 0.0

    bp_max_iter = int(Hx.shape[1] / 10)
    Lx, Lz = get_logical_operators_by_pivoting(Hx, Hz)

    bp_osd_decoder = BpOsdDecoder(
        pcm=Hz,
        error_rate=float(p),
        max_iter=bp_max_iter,
        bp_method='minimum_sum',
        ms_scaling_factor=0.625,
        schedule='parallel',
        osd_method='OSD_CS',
        osd_order=2,
    )

    ler, stderr, runtime = compute_logical_error_rate(
        Hz, Lz, p, run_count=budget, DECODER=bp_osd_decoder,
        run_label=run_label, DEBUG=False
    )
    return ler, stderr, runtime


# ============================================================
# HDF5 HELPERS
# ============================================================

def _ensure_ds(grp, name, shape_sample, dtype):
    if name in grp:
        return grp[name]
    shape = (0,) + shape_sample
    maxshape = (None,) + shape_sample
    return grp.create_dataset(
        name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True
    )


def append_to_hdf5(grp, edge_list, params, ler=0.0, std=0.0, runtime=0.0,
                   logical_weight=0.0, accepted=False, step=-1, trial=-1):
    """
    Append a visited state and metadata.
    """
    ds_states = _ensure_ds(grp, "states", (edge_list.shape[0],), np.uint32)
    ds_ler = _ensure_ds(grp, "logical_error_rates", (), np.float64)
    ds_std = _ensure_ds(grp, "logical_error_rates_std", (), np.float64)
    ds_dcl = _ensure_ds(grp, "distances_classical", (), np.float64)
    ds_dclT = _ensure_ds(grp, "distances_classical_T", (), np.float64)
    ds_dq = _ensure_ds(grp, "distances_quantum", (), np.float64)
    ds_run = _ensure_ds(grp, "decoding_runtimes", (), np.float64)

    ds_logw = _ensure_ds(grp, "target_logical_weight", (), np.float64)
    ds_acc = _ensure_ds(grp, "accepted", (), np.uint8)
    ds_step = _ensure_ds(grp, "search_step", (), np.int32)
    ds_trial = _ensure_ds(grp, "search_trial", (), np.int32)

    idx = ds_ler.shape[0]

    ds_states.resize(idx + 1, axis=0)
    ds_states[idx] = edge_list

    ds_ler.resize(idx + 1, axis=0)
    ds_ler[idx] = ler

    ds_std.resize(idx + 1, axis=0)
    ds_std[idx] = std

    ds_dcl.resize(idx + 1, axis=0)
    ds_dcl[idx] = params["d_classical"]

    ds_dclT.resize(idx + 1, axis=0)
    ds_dclT[idx] = params["d_T_classical"]

    ds_dq.resize(idx + 1, axis=0)
    ds_dq[idx] = params["d_quantum"]

    ds_run.resize(idx + 1, axis=0)
    ds_run[idx] = runtime

    ds_logw.resize(idx + 1, axis=0)
    ds_logw[idx] = logical_weight

    ds_acc.resize(idx + 1, axis=0)
    ds_acc[idx] = 1 if accepted else 0

    ds_step.resize(idx + 1, axis=0)
    ds_step[idx] = step

    ds_trial.resize(idx + 1, axis=0)
    ds_trial[idx] = trial

    return idx


def update_hdf5_row(grp, idx, ler, std, runtime):
    grp["logical_error_rates"][idx] = ler
    grp["logical_error_rates_std"][idx] = std
    grp["decoding_runtimes"][idx] += runtime


# ============================================================
# LOGICAL-GUIDED SEARCH HELPERS
# ============================================================

def get_variable_nodes(G: nx.MultiGraph) -> list[int]:
    return sorted([n for n, b in G.nodes(data="bipartite") if b == 1])


def get_check_nodes(G: nx.MultiGraph) -> list[int]:
    return sorted([n for n, b in G.nodes(data="bipartite") if b == 0])


def get_lowest_weight_logical(logicals: np.ndarray) -> tuple[np.ndarray, int]:
    if logicals.shape[0] == 0:
        raise ValueError("No logical operators found.")
    weights = np.sum(logicals, axis=1)
    idx = int(np.argmin(weights))
    return logicals[idx], int(weights[idx])


def propose_targeted_swap_from_logical(
    state: nx.MultiGraph,
    logical_support_cols: np.ndarray,
    max_tries: int = 200,
    tried_proposals: set | None = None,
):
    support_vars = support_indices_to_graph_nodes(state, logical_support_cols)
    non_support_vars = set(get_variable_nodes(state)) - support_vars

    if not support_vars or not non_support_vars:
        return None

    bad_edges = []
    for v in support_vars:
        for c in state.neighbors(v):
            bad_edges.append((c, v))

    if not bad_edges:
        return None

    checks_touching_support = set()
    for v in support_vars:
        checks_touching_support.update(state.neighbors(v))

    clean_edges = []
    for c in get_check_nodes(state):
        if c in checks_touching_support:
            continue
        for v in state.neighbors(c):
            if v in non_support_vars:
                clean_edges.append((c, v))

    if not clean_edges:
        return None

    for _ in range(max_tries):
        e1 = random.choice(bad_edges)
        e2 = random.choice(clean_edges)

        c1, v1 = e1
        c2, v2 = e2

        if c1 == c2 or v1 == v2:
            continue

        f1 = (c1, v2)
        f2 = (c2, v1)

        if state.has_edge(*f1) or state.has_edge(*f2):
            continue

        edges_to_remove = [e1, e2]
        edges_to_add = [f1, f2]

        proposal_key = canonicalize_proposal(edges_to_add, edges_to_remove)

        if tried_proposals is not None and proposal_key in tried_proposals:
            continue

        return edges_to_add, edges_to_remove

    return None

from itertools import combinations

def find_low_weight_classical_codeword(H_csr, start_weight=1, max_weight=None):
    """
    Brute-force toy version.
    For larger codes, replace with a smarter search / estimate.
    Returns a binary vector in ker(H) with low weight.
    """
    H = H_csr.toarray().astype(np.uint8)
    n = H.shape[1]

    if max_weight is None:
        max_weight = n

    for w in range(start_weight, max_weight + 1):
        for cols in combinations(range(n), w):
            e = np.zeros(n, dtype=np.uint8)
            e[list(cols)] = 1
            syndrome = (H @ e) % 2
            if np.all(syndrome == 0):
                return e, w

    return None, None

def is_classical_support_detectable(H: np.ndarray, support: np.ndarray) -> bool:
    e = np.zeros(H.shape[1], dtype=np.uint8)
    e[support] = 1
    syndrome = (H @ e) % 2
    return np.any(syndrome != 0)

def canonicalize_proposal(edges_to_add, edges_to_remove):
    add_key = tuple(sorted(tuple(edge) for edge in edges_to_add))
    remove_key = tuple(sorted(tuple(edge) for edge in edges_to_remove))
    return (add_key, remove_key)


def improve_state_by_breaking_low_weight_logical(
    state: nx.MultiGraph,
    max_trials: int = 100,
    require_distance_non_decrease: bool = True,
    verbose: bool = True,
):
    params, Hx, Hz = get_code_parameters_and_matrices(state)
    current_distance = min(params["d_classical"], params["d_T_classical"])

    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)

    logical_vec, logical_weight = find_low_weight_classical_codeword(
        csr_H,
        start_weight=1,
        max_weight=min(10, csr_H.shape[1])
    )

    if logical_vec is None:
        if verbose:
            print("  No low-weight classical codeword found.")
        return {
            "accepted": False,
            "state": state,
            "params": params,
            "logical_weight": -1,
            "trial": -1,
            "edges_to_add": None,
            "edges_to_remove": None,
            "distance_before": current_distance,
            "distance_after": current_distance,
        }

    support = np.where(logical_vec == 1)[0]

    if verbose:
        print(f"  Target logical weight: {logical_weight}")
        print(f"  Current distance: {current_distance}")

    tried_proposals = set()
    best_attempts = []

    for trial in range(max_trials):
        proposal = propose_targeted_swap_from_logical(state, support, max_tries=100, tried_proposals=tried_proposals)
        
        if proposal is None:
            if verbose:
                print(f"  No valid proposal found at trial {trial + 1} /{max_trials}.")
            break
        
        edges_to_add, edges_to_remove = proposal
        proposal_key = canonicalize_proposal(edges_to_add, edges_to_remove)
        tried_proposals.add(proposal_key)

        new_state = add_and_remove_edges(state, edges_to_add, edges_to_remove)

        new_params, Hx_new, Hz_new = get_code_parameters_and_matrices(new_state)

        H_new = tanner_graph_to_parity_check_matrix(new_state)
        if not is_classical_support_detectable(H_new, support):
            print(f"  Rejected at trial {trial + 1} /{max_trials} due to undetectable support.")
            continue

        new_distance = min(new_params["d_classical"], new_params["d_T_classical"])

        if require_distance_non_decrease and new_distance < current_distance:
            continue

        best_attempts.append({
            "state": new_state,
            "params": new_params,
            "logical_weight": logical_weight,
            "trial": trial,
            "edges_to_add": edges_to_add,
            "edges_to_remove": edges_to_remove,
            "distance_before": current_distance,
            "distance_after": new_distance,
        })

        if new_distance > current_distance:
            print(f"  Accepted at trial {trial + 1} /{max_trials} with distance improvement: {current_distance} -> {new_distance}")
            return {"accepted": True, **best_attempts[-1]}

    if best_attempts:
        return {"accepted": True, **best_attempts[0]}

    return {
        "accepted": False,
        "state": state,
        "params": params,
        "logical_weight": logical_weight,
        "trial": -1,
        "edges_to_add": None,
        "edges_to_remove": None,
        "distance_before": current_distance,
        "distance_after": current_distance,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", default=0, type=int, help="Code family index")
    parser.add_argument("-p", default=None, type=float, help="Physical error rate")
    parser.add_argument("-S", default=SEARCH_STEPS, type=int, help="Search steps")
    parser.add_argument("-T", default=TRIALS_PER_STEP, type=int, help="Trials per step")
    parser.add_argument("--screen_budget", default=BUDGET_SCREENING, type=int)
    parser.add_argument("--prec_budget", default=BUDGET_PRECISION, type=int)
    parser.add_argument("--topk", default=TOP_K_PRECISION, type=int)
    args = parser.parse_args()

    C = args.C
    p = noise_levels[C] if args.p is None else args.p

    print("\n--- LOGICAL GUIDED SEARCH ---")
    print(f"Code family: {codes[C]}")
    print(f"Noise level p = {p}")
    print(f"Search steps = {args.S}")
    print(f"Trials/step = {args.T}")
    print(f"Budgets: screening={args.screen_budget}, precision={args.prec_budget}")

    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])
    start_time = time.time()

    with h5py.File(output_file, "a") as f:
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

        # Save initial state
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
        )

        all_candidates = [{
            "state": current_state,
            "params": params_init,
            "row_idx": idx_init,
            "dist": min(params_init["d_classical"], params_init["d_T_classical"]),
            "logical_weight": 0.0,
        }]

        global_max_dist = all_candidates[0]["dist"]

        # -------------------------
        # Phase 1: logical-guided search
        # -------------------------
        print("\n>>> PHASE 1: Logical-guided search")

        for step in range(1, args.S + 1):
            print(f"\nStep {step}/{args.S} (global max dist so far = {global_max_dist})")

            result = improve_state_by_breaking_low_weight_logical(
                state=current_state,
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

        # -------------------------
        # Phase 2: screening
        # -------------------------
        print(f"\n>>> PHASE 2: Screening ({args.screen_budget})")

        target_dist = global_max_dist
        screening_candidates = [c for c in all_candidates if c["dist"] == target_dist]
        unique_candidates_dict = {}
        for c in screening_candidates:
            edge_key = tuple(parse_edgelist(c["state"]).flatten().tolist())
            if edge_key not in unique_candidates_dict:
                unique_candidates_dict[edge_key] = c

        unique_candidates = list(unique_candidates_dict.values())

        print(f"Found {len(all_candidates)} total saved states.")
        print(f"Screening {len(unique_candidates)} unique candidates with dist == {target_dist}")

        screened_results = []

        for cand in tqdm(unique_candidates, desc="Running MC (screening)"):
            params, Hx, Hz = get_code_parameters_and_matrices(cand["state"])
            ler, std, run_t = evaluate_mc(
                Hx, Hz, p, args.screen_budget,
                run_label=f"screen_row_{cand['row_idx']}"
            )

            update_hdf5_row(run_grp, cand["row_idx"], ler, std, run_t)

            cand["ler"] = ler
            cand["std"] = std
            cand["runtime"] = run_t
            screened_results.append(cand)
            f.flush()

        if not screened_results:
            print("No candidates to screen.")
            sys.exit()

        screened_results.sort(key=lambda x: x["ler"])
        print(f"Best screening LER: {screened_results[0]['ler']:.6f} (dist={screened_results[0]['dist']})")

        # -------------------------
        # Phase 3: precision
        # -------------------------
        print(f"\n>>> PHASE 3: Precision ({args.prec_budget})")

        top_candidates = screened_results[:args.topk]
        print(f"Promoting top {len(top_candidates)} candidates")

        final_best_cand = None
        min_final_ler = np.inf
        final_best_std = np.inf

        for cand in top_candidates:
            print(f"Evaluating row {cand['row_idx']} | dist={cand['dist']} | screening LER={cand['ler']:.6f}")
            params, Hx, Hz = get_code_parameters_and_matrices(cand["state"])

            ler, std, run_t = evaluate_mc(
                Hx, Hz, p, args.prec_budget,
                run_label=f"precision_row_{cand['row_idx']}"
            )

            update_hdf5_row(run_grp, cand["row_idx"], ler, std, run_t)
            print(f"  -> LER: {ler:.6f} ± {std:.6f} | runtime={run_t:.2f}s")

            if ler < min_final_ler:
                min_final_ler = ler
                final_best_std = std
                final_best_cand = cand

            f.flush()

        # -------------------------
        # Save best state
        # -------------------------
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
                dtype=np.uint32
            )
            run_grp.attrs["min_cost"] = min_final_ler
            run_grp.attrs["best_dist"] = final_best_cand["dist"]

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal time: {total_time/3600:.2f} hours")
    print(f"Saved to {output_file}")