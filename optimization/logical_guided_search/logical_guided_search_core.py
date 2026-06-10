import random
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Optional

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from itertools import combinations
from ldpc import mod2

from optimization.experiments_settings import (
    add_and_remove_edges,
    tanner_graph_to_parity_check_matrix,
    parse_edgelist,
)


def format_score_info(score_info):
    if score_info is None:
        return "weight_patterns={}"

    components = score_info.get("components", {})
    components = {int(k): int(v) for k, v in components.items()}

    return (
        f"weight_patterns={components}, "
        f"score={float(score_info.get('score', np.nan)):.6g}"
    )


def get_A_dq(cand):
    d = int(cand["dist"])
    score_info = cand.get("score_info", {})
    components = score_info.get("components", {})
    return int(components.get(d, 10**18))


def beam_rank_key(cand):
    """
    Absolute-score ranking.

    Lower low_weight_score is better.
    Distance is used only as a tie-breaker.
    """
    return (
        float(cand.get("low_weight_score", np.inf)),
        -int(cand.get("dist", -1)),
        float(cand.get("logical_weight", np.inf)),
        int(cand.get("row_idx", 10**18)),
    )


def get_variable_nodes(G: nx.MultiGraph) -> list[int]:
    return sorted([n for n, b in G.nodes(data="bipartite") if b == 1])


def get_check_nodes(G: nx.MultiGraph) -> list[int]:
    return sorted([n for n, b in G.nodes(data="bipartite") if b == 0])


def support_indices_to_graph_nodes(G: nx.MultiGraph, support: np.ndarray) -> set[int]:
    variable_nodes = get_variable_nodes(G)
    if np.max(support, initial=-1) >= len(variable_nodes):
        raise ValueError(
            f"Support index out of range: max support index={np.max(support)}, "
            f"num variable nodes={len(variable_nodes)}"
        )
    return {variable_nodes[i] for i in support}


def canonicalize_proposal(edges_to_add, edges_to_remove):
    add_key = tuple(sorted(tuple(edge) for edge in edges_to_add))
    remove_key = tuple(sorted(tuple(edge) for edge in edges_to_remove))
    return add_key, remove_key


def propose_targeted_swap_from_logical(
    state: nx.MultiGraph,
    logical_support_cols: np.ndarray,
    max_tries: int = 200,
    tried_proposals: Optional[set] = None,
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

        # if state.has_edge(*f1):
        #     print(f"WARNING: proposed edge already exists in state, losing one edge: f1={f1}")
        #     # continue
        # if state.has_edge(*f2):
        #     print(f"WARNING: proposed edge already exists in state, losing one edge: f2={f2}")
        #     # continue

        edges_to_remove = [e1, e2]
        edges_to_add = [f1, f2]

        proposal_key = canonicalize_proposal(edges_to_add, edges_to_remove)
        if tried_proposals is not None and proposal_key in tried_proposals:
            continue

        return edges_to_add, edges_to_remove

    return None


def generate_logical_guided_candidates(
    state: nx.MultiGraph,
    get_code_parameters_and_matrices,
    max_trials: int = 100,
    logical_max_comb_order: int = 5,
    weight_slack: int = 1,
    require_detectable: bool = True,
    require_distance_non_decrease: bool = True,
    verbose: bool = False,
    seen_keys: Optional[set] = None,
    score_candidate_fn=None,
):
    """
    Generate multiple logical-guided child states from one parent state.

    This uses almost the same logic as improve_state_by_breaking_low_weight_logical,
    but returns the top max_candidates valid moves instead of only one best move.
    """
    params, _, _ = get_code_parameters_and_matrices(state)
    current_distance = min(params["d_classical"], params["d_T_classical"])

    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)

    logical_words = find_low_weight_classical_codewords(
        csr_H,
        max_comb_order=logical_max_comb_order,
        max_words=30,
        weight_slack=weight_slack,
    )

    print(f"Found {len(logical_words)} low-weight logical codewords with max_comb_order={logical_max_comb_order}.")

    if not logical_words:
        return []

    if verbose:
        print(f"  Current distance: {current_distance}")
        print(
            f"  Generating candidates with max_trials={max_trials}, "
            f"logical_max_comb_order={logical_max_comb_order}..."
        )

    tried_proposals = set()
    valid_attempts = []
    reject_count = 0
    skipped_seen = 0

    for trial in range(max_trials):
        logical_weight, logical_vec = random.choice(logical_words)
        support = np.where(logical_vec == 1)[0]
        proposal = propose_targeted_swap_from_logical(
            state,
            support,
            max_tries=100,
            tried_proposals=tried_proposals,
        )

        if proposal is None:
            if verbose:
                print(
                    f"  No valid proposal found at trial {trial + 1}/{max_trials}.")
            continue

        edges_to_add, edges_to_remove = proposal
        proposal_key = canonicalize_proposal(edges_to_add, edges_to_remove)
        tried_proposals.add(proposal_key)

        new_state = add_and_remove_edges(state, edges_to_add, edges_to_remove)

        try:
            new_key = tuple(parse_edgelist(new_state).flatten().tolist())
        except Exception:
            new_key = None

        if seen_keys is not None and new_key is not None and new_key in seen_keys:
            skipped_seen += 1
            continue

        new_params, _, _ = get_code_parameters_and_matrices(new_state)

        H_new = tanner_graph_to_parity_check_matrix(new_state)
        if require_detectable and not is_classical_support_detectable(H_new, support):
            if verbose:
                print(
                    f"  Rejected at trial {trial + 1}/{max_trials}: support still undetectable.")
            continue

        new_distance = min(new_params["d_classical"],
                           new_params["d_T_classical"])

        if require_distance_non_decrease and new_distance < current_distance:
            reject_count += 1
            if verbose:
                print(
                    f"  Rejected at trial {trial + 1}/{max_trials}: "
                    f"distance {current_distance}->{new_distance}."
                )
            continue

        score_info = None
        low_weight_score = np.inf

        if score_candidate_fn is not None:
            try:
                score_info = score_candidate_fn(new_state, new_params)
                low_weight_score = float(score_info["score"])
            except Exception as exc:
                if verbose:
                    print(f"  WARNING: score computation failed: {exc}")
                low_weight_score = np.inf

        cand = {
            "state": new_state,
            "params": new_params,
            "logical_weight": logical_weight,
            "support": support.copy(),
            "trial": trial,
            "edges_to_add": edges_to_add,
            "edges_to_remove": edges_to_remove,
            "distance_before": current_distance,
            "distance_after": new_distance,
            "dist": new_distance,
            "low_weight_score": low_weight_score,
            "score_info": score_info,
        }

        valid_attempts.append(cand)

        if verbose:
            print(
                f"  Valid proposal at trial {trial + 1}/{max_trials}: "
                f"distance {current_distance}->{new_distance}, "
                f"target_weight_before_swap={logical_weight}, "
                f"{format_score_info(score_info)}"
            )

    if verbose:
        trials_done = trial + 1 if "trial" in locals() else 0
        print(
            f"  Trials completed: {trials_done}/{max_trials}, "
            f"valid proposals found: {len(valid_attempts)}, "
            f"skipped seen: {skipped_seen}, "
            f"rejections due to distance decrease: {reject_count}."
        )

    if not valid_attempts:
        return []

    valid_attempts.sort(key=beam_rank_key)
    return valid_attempts


def find_low_weight_classical_codewords(
    H,
    max_comb_order=None,
    max_words=20,
    weight_slack=0,
):
    """
    Return several low-weight codewords in ker(H), not just one.
    weight_slack=0 means only minimum-weight found codewords.
    weight_slack=1 means include min_weight and min_weight+1.
    """
    ker = mod2.nullspace(H).toarray().astype(np.uint8)

    if ker.shape[0] == 0:
        return []

    num_basis = ker.shape[0]
    max_comb_order = min(max_comb_order, num_basis) if max_comb_order is not None else num_basis

    found = []
    seen = set()
    best_weight = np.inf

    def add_vec(v):
        nonlocal best_weight
        w = int(v.sum())
        if w == 0:
            return

        key = tuple(np.flatnonzero(v).tolist())
        if key in seen:
            return

        seen.add(key)
        best_weight = min(best_weight, w)
        found.append((w, v.copy()))

    for i in range(num_basis):
        add_vec(ker[i])

    for r in range(2, max_comb_order + 1):
        for idxs in combinations(range(num_basis), r):
            v = np.bitwise_xor.reduce(ker[list(idxs)], axis=0).astype(np.uint8)
            add_vec(v)

    if not found:
        return []

    cutoff = best_weight + weight_slack
    found = [(w, v) for (w, v) in found if w <= cutoff]
    random.shuffle(found)
    found.sort(key=lambda x: x[0])  # stable sort, keeps random order inside same weight

    return found[:max_words]


def is_classical_support_detectable(H: np.ndarray, support: np.ndarray) -> bool:
    e = np.zeros(H.shape[1], dtype=np.uint8)
    e[support] = 1
    syndrome = (H @ e) % 2
    return np.any(syndrome != 0)


def improve_state_by_breaking_low_weight_logical(
    state: nx.MultiGraph,
    get_code_parameters_and_matrices,
    max_trials: int = 100,
    logical_max_comb_order: int = 10,
    require_distance_non_decrease: bool = True,
    verbose: bool = True,
    score_candidate_fn=None,
):
    params, _, _ = get_code_parameters_and_matrices(state)
    current_distance = min(params["d_classical"], params["d_T_classical"])

    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)

    logical_words = find_low_weight_classical_codewords(
    csr_H,
    max_comb_order=logical_max_comb_order,
    max_words=20,
    weight_slack=0,
    )

    if not logical_words:
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
            "low_weight_score": np.inf,
            "score_info": None,
        }

    logical_weight, logical_vec = random.choice(logical_words)
    support = np.where(logical_vec == 1)[0]

    if verbose:
        print(f"  Target logical weight: {logical_weight}")
        print(f"  Current distance: {current_distance}")
        print(
            f"  Generating candidates with max_trials={max_trials} and logical_max_comb_order={logical_max_comb_order}...")

    tried_proposals = set()
    valid_attempts = []
    reject_count = 0

    for trial in range(max_trials):
        proposal = propose_targeted_swap_from_logical(
            state,
            support,
            max_tries=100,
            tried_proposals=tried_proposals,
        )

        if proposal is None:
            if verbose:
                print(
                    f"  No valid unseen proposal found at trial {trial + 1}/{max_trials}.")
            break

        edges_to_add, edges_to_remove = proposal
        proposal_key = canonicalize_proposal(edges_to_add, edges_to_remove)
        tried_proposals.add(proposal_key)

        new_state = add_and_remove_edges(state, edges_to_add, edges_to_remove)
        new_params, _, _ = get_code_parameters_and_matrices(new_state)

        H_new = tanner_graph_to_parity_check_matrix(new_state)
        if not is_classical_support_detectable(H_new, support):
            if verbose:
                print(
                    f"  Rejected at trial {trial + 1}/{max_trials}: support still undetectable.")
            continue

        new_distance = min(new_params["d_classical"],
                           new_params["d_T_classical"])

        if require_distance_non_decrease and new_distance < current_distance:
            reject_count += 1
            if verbose:
                print(
                    f"  Rejected at trial {trial + 1}/{max_trials}: "
                    f"distance {current_distance}->{new_distance}."
                )
            continue

        score_info = None
        low_weight_score = np.inf

        if score_candidate_fn is not None:
            try:
                score_info = score_candidate_fn(new_state, new_params)
                low_weight_score = float(score_info["score"])
            except Exception as exc:
                if verbose:
                    print(f"  WARNING: score computation failed: {exc}")
                low_weight_score = np.inf

        if verbose:
            print(
                f"  Valid proposal at trial {trial + 1}/{max_trials}: "
                f"distance {current_distance}->{new_distance}, "
                f"target_weight_before_swap={logical_weight}, "
                f"{format_score_info(score_info)}"
            )

        valid_attempts.append(
            {
                "state": new_state,
                "params": new_params,
                "logical_weight": logical_weight,
                "trial": trial,
                "edges_to_add": edges_to_add,
                "edges_to_remove": edges_to_remove,
                "distance_before": current_distance,
                "distance_after": new_distance,
                "low_weight_score": low_weight_score,
                "score_info": score_info,
                "dist": new_distance,
                "support": support.copy(),
            }
        )

    if verbose:
        print(
            f"  Trials completed: {trial + 1}/{max_trials}, "
            f"valid proposals found: {len(valid_attempts)}, "
            f"rejections due to distance decrease: {reject_count}."
        )

    if valid_attempts:
        # Main rule:
        #   1. maximize distance
        #   2. minimize weighted low-weight score
        valid_attempts.sort(key=beam_rank_key)

        best = valid_attempts[0]

        if verbose:
            print(
                "  Accepted best valid move: "
                f"distance {best['distance_before']}->{best['distance_after']}, "
                f"target_weight_before_swap={best['logical_weight']}, "
                f"weight_patterns={format_score_info(best.get('score_info'))}, "
                f"score={best['low_weight_score']:.6g}"
            )

        return {"accepted": True, **best}

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
        "low_weight_score": np.inf,
        "score_info": None,
    }


def run_logical_guided_search(
    initial_state: nx.MultiGraph,
    get_code_parameters_and_matrices,
    steps: int = 10,
    max_trials: int = 100,
    verbose: bool = True,
):
    state = nx.MultiGraph(initial_state)
    history = []

    for step in range(steps):
        if verbose:
            print(f"\n=== Step {step} ===")

        result = improve_state_by_breaking_low_weight_logical(
            state=state,
            get_code_parameters_and_matrices=get_code_parameters_and_matrices,
            max_trials=max_trials,
            require_distance_non_decrease=True,
            verbose=verbose,
        )
        history.append(result)

        if not result["accepted"]:
            break

        state = result["state"]

    return state, history
