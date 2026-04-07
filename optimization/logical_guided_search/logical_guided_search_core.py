import random
from itertools import combinations
from typing import Optional

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from optimization.experiments_settings import (
    add_and_remove_edges,
    tanner_graph_to_parity_check_matrix,
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

        if state.has_edge(*f1) or state.has_edge(*f2):
            continue

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
    num_candidates: int = 20,
    proposal_max_tries: int = 300,
    logical_max_comb_order: int = 5,
    require_detectable: bool = True,
    require_distance_non_decrease: bool = False,
    verbose: bool = False,
):
    """
    Generate multiple logical-guided child states from one parent state.
    This is for beam search expansion.

    Returns
    -------
    candidates : list[dict]
        Each dict contains:
        {
            "state",
            "params",
            "logical_weight",
            "support",
            "edges_to_add",
            "edges_to_remove",
            "distance_before",
            "distance_after",
        }
    """
    params, _, _ = get_code_parameters_and_matrices(state)
    current_distance = min(params["d_classical"], params["d_T_classical"])

    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)

    logical_vec, logical_weight = find_low_weight_classical_codeword(
        csr_H,
        max_comb_order=logical_max_comb_order,
    )

    if logical_vec is None:
        if verbose:
            print("  No low-weight classical codeword found.")
        return []

    support = np.where(logical_vec == 1)[0]

    if verbose:
        print(f"  Target logical weight: {logical_weight}")
        print(f"  Current distance: {current_distance}")

    tried_proposals = set()
    candidates = []

    cand_trial_idx = 1
    while len(candidates) < num_candidates:
        proposal = propose_targeted_swap_from_logical(
            state,
            support,
            max_tries=proposal_max_tries,
            tried_proposals=tried_proposals,
        )

        if proposal is None:
            break

        edges_to_add, edges_to_remove = proposal
        proposal_key = canonicalize_proposal(edges_to_add, edges_to_remove)
        tried_proposals.add(proposal_key)

        new_state = add_and_remove_edges(state, edges_to_add, edges_to_remove)
        new_params, _, _ = get_code_parameters_and_matrices(new_state)

        H_new = tanner_graph_to_parity_check_matrix(new_state)
        if require_detectable and not is_classical_support_detectable(H_new, support):
            if verbose:
                print(f"  Rejected at {cand_trial_idx}/{num_candidates}: support still undetectable.")
            continue

        new_distance = min(new_params["d_classical"], new_params["d_T_classical"])

        if require_distance_non_decrease and new_distance < current_distance:
            if verbose:
                print(f"  Rejected at {cand_trial_idx}/{num_candidates}: distance {current_distance}->{new_distance}")
            continue

        candidates.append({
            "state": new_state,
            "params": new_params,
            "logical_weight": logical_weight,
            "support": support.copy(),
            "edges_to_add": edges_to_add,
            "edges_to_remove": edges_to_remove,
            "distance_before": current_distance,
            "distance_after": new_distance,
        })

        cand_trial_idx += 1

    return candidates


import numpy as np
from itertools import combinations
from ldpc import mod2


def find_low_weight_classical_codeword(H, max_comb_order=None):
    """
    Find a low-weight nonzero classical codeword in ker(H)
    by searching linear combinations of kernel basis vectors.

    Parameters
    ----------
    H : np.ndarray or csr_matrix
        Classical parity-check matrix.
    max_comb_order : int or None
        Maximum number of basis vectors to combine.
        If None, searches all.

    Returns
    -------
    best_vec : np.ndarray or None
        Lowest-weight codeword found.
    best_weight : int or None
        Its Hamming weight.
    """
    ker = mod2.nullspace(H).toarray().astype(np.uint8)

    if ker.shape[0] == 0:
        return None, None

    num_basis = ker.shape[0]
    if max_comb_order is None:
        max_comb_order = num_basis

    best_vec = None
    best_weight = np.inf

    # Also check single basis vectors first
    for i in range(num_basis):
        v = ker[i].copy()
        w = int(v.sum())
        if 0 < w < best_weight:
            best_vec = v
            best_weight = w

    # Check combinations of basis vectors
    for r in range(2, max_comb_order + 1):
        for idxs in combinations(range(num_basis), r):
            v = np.bitwise_xor.reduce(ker[list(idxs)], axis=0).astype(np.uint8)
            w = int(v.sum())
            if 0 < w < best_weight:
                best_vec = v
                best_weight = w

    if best_vec is None:
        return None, None

    return best_vec, best_weight


def is_classical_support_detectable(H: np.ndarray, support: np.ndarray) -> bool:
    e = np.zeros(H.shape[1], dtype=np.uint8)
    e[support] = 1
    syndrome = (H @ e) % 2
    return np.any(syndrome != 0)


def improve_state_by_breaking_low_weight_logical(
    state: nx.MultiGraph,
    get_code_parameters_and_matrices,
    max_trials: int = 100,
    require_distance_non_decrease: bool = True,
    verbose: bool = True,
):
    params, _, _ = get_code_parameters_and_matrices(state)
    current_distance = min(params["d_classical"], params["d_T_classical"])

    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)

    logical_vec, logical_weight = find_low_weight_classical_codeword(
        csr_H,
        max_comb_order=5,
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
        proposal = propose_targeted_swap_from_logical(
            state,
            support,
            max_tries=100,
            tried_proposals=tried_proposals,
        )

        if proposal is None:
            if verbose:
                print(f"  No valid unseen proposal found at trial {trial + 1}/{max_trials}.")
            break

        edges_to_add, edges_to_remove = proposal
        proposal_key = canonicalize_proposal(edges_to_add, edges_to_remove)
        tried_proposals.add(proposal_key)

        new_state = add_and_remove_edges(state, edges_to_add, edges_to_remove)
        new_params, _, _ = get_code_parameters_and_matrices(new_state)

        H_new = tanner_graph_to_parity_check_matrix(new_state)
        if not is_classical_support_detectable(H_new, support):
            if verbose:
                print(f"  Rejected at trial {trial + 1}/{max_trials}: support still undetectable.")
            continue

        new_distance = min(new_params["d_classical"], new_params["d_T_classical"])

        if require_distance_non_decrease and new_distance < current_distance:
            if verbose:
                print(f"  Rejected at trial {trial + 1}/{max_trials}: distance {current_distance}->{new_distance}.")
            continue
        
        # Record this valid attempt, even if it doesn't improve distance, to allow accepting the first non-decreasing move if no improving move is found.
        print(f"  Valid proposal at trial {trial + 1}/{max_trials}: distance {current_distance}->{new_distance}.")
        attempt = {
            "state": new_state,
            "params": new_params,
            "logical_weight": logical_weight,
            "trial": trial,
            "edges_to_add": edges_to_add,
            "edges_to_remove": edges_to_remove,
            "distance_before": current_distance,
            "distance_after": new_distance,
        }
        best_attempts.append(attempt)

        # Accept immediately if it improves distance
        if new_distance > current_distance:
            if verbose:
                print(
                    f"  Accepted at trial {trial + 1}/{max_trials}: "
                    f"distance {current_distance}->{new_distance}"
                )
            return {"accepted": True, **attempt}

    if best_attempts:
        if verbose:
            print("  Accepted first non-decreasing valid move.")
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