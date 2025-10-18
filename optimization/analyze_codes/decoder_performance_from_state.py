import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basic_css_code import construct_HGP_code
from optimization.experiments_settings import tanner_graph_to_parity_check_matrix
import numpy as np

import networkx as nx
from scipy.sparse import csr_matrix

import ldpc
import ldpc.code_util
from ldpc.bposd_decoder import BpOsdDecoder
from logical_operators import get_logical_operators_by_pivoting
from decoder_performance import compute_logical_error_rate

DISTANCE_THRESHOLD = 8  # minimum distance threshold to run the decoder performance evaluation

def compute_classical_code_parameters(H: csr_matrix) -> tuple[int, int, int]:
    """Compute the parameters [n, k, d] of a classical linear code given its parity-check matrix H.

    Args:
        H (csr_matrix): The parity-check matrix of the code.

    Returns:
        tuple[int, int, int]: A tuple containing the parameters (n, k, d) where
            n is the length of the code,
            k is the dimension of the code,
            d is the minimum distance of the code.
    """
    n = H.shape[1]
    r = ldpc.mod2.rank(H)
    k = n - r
    d = ldpc.code_util.compute_exact_code_distance(H)
    return n, k, d

def compute_hgp_code_distance_lower_bound(H: csr_matrix, HT: csr_matrix) -> int:
    """Compute the lower bound on the distance of the hypergraph product code constructed from H and H^T.

    Args:
        H (csr_matrix): The parity-check matrix of the code.
        HT (csr_matrix): The transpose of the parity-check matrix H.

    Returns:
        int: The lower bound on the distance of the hypergraph product code.
    """
    r_classical = ldpc.mod2.rank(H)
    _, _, d = compute_classical_code_parameters(H)
    if r_classical == H.shape[0]:
        # H is full rank, so we skip computing the parameters of H^T
        return d
    else:
        _, _, dT = compute_classical_code_parameters(HT)
        return min(d, dT)

def evaluate_performance_of_state(state: nx.MultiGraph, p_vals: np.ndarray, MC_budget: int, bp_max_iter=None, run_label="Best neighbor search", distance_threshold=DISTANCE_THRESHOLD, canskip=True) -> dict:
    """
    Evaluate the decoding performance (logical error rates) of a given state.
    Parameters:
        state (nx.MultiGraph): The Tanner graph representing the code.
        p_vals (np.ndarray): Array of physical error rates to evaluate.
        MC_budget (int): The number of Monte Carlo runs to perform for each error rate.
        bp_max_iter (int): Maximum number of iterations for the BP decoder.
    Returns:
        logical_error_rates (list): List of logical error rates for each physical error rate.
        stds (list): List of standard deviations of the logical error rates.
        runtimes (list): List of runtimes for each decoding operation.
    """
    H = tanner_graph_to_parity_check_matrix(state)

    csr_H = csr_matrix(H, dtype=np.uint8)
    r_classical = ldpc.mod2.rank(csr_H)
    
    # n_classical, k_classical, d_classical = ldpc.code_util.compute_code_parameters(csr_H)
    n_classical, k_classical, _ = ldpc.code_util.compute_code_parameters(csr_H)
    d_classical = ldpc.code_util.compute_exact_code_distance(csr_H)
    print(f"H: [{n_classical}, {k_classical}, {d_classical}]")
    if k_classical == n_classical - csr_H.shape[0]:
        n_T_classical = csr_H.shape[0]
        k_T_classical = n_T_classical - r_classical # k will be 0 if full rank
        d_T_classical = np.inf
    else:
        n_T_classical, k_T_classical, d_T_classical = ldpc.code_util.compute_code_parameters(csr_matrix(H.T, dtype=np.uint8))
    print(f"H^T: [{n_T_classical}, {k_T_classical}, {d_T_classical}]")

    Hx, Hz = construct_HGP_code(H)

    n_quantum = n_classical**2 + n_T_classical**2
    k_quantum = k_classical**2 + k_T_classical**2
    d_quantum = min(d_classical, d_T_classical)
    print(f"Q: [[{n_quantum}, {k_quantum}, {d_quantum}]]")

    base_payload = {
        "n_classical": n_classical,
        "k_classical": k_classical,
        "d_classical": d_classical,
        "n_T_classical": n_T_classical,
        "k_T_classical": k_T_classical,
        "d_T_classical": d_T_classical,
        "rank_H": r_classical,
        "n_quantum": n_quantum,
        "k_quantum": k_quantum,
        "d_quantum": d_quantum,
        "d_Hx": 0,
        "d_Hz": 0
    }

    min_classical_distance = min(d_classical, d_T_classical)
    if min_classical_distance < distance_threshold and canskip:
        print(f"Distance {min_classical_distance} is below threshold {distance_threshold}. Skipping performance evaluation.")
        # save placeholders
        return {
            "logical_error_rates": [0.0]*len(p_vals),
            "stderrs": [0.0]*len(p_vals),
            **base_payload,
            "runtimes": [0.0]*len(p_vals),
            "skipped": True
        }

    if bp_max_iter is None:
        bp_max_iter = int(Hx.shape[1]/10)
    
    # osd_order = 60
    osd_order = 2
    ms_scaling_factor = 0.625

    Lx, Lz = get_logical_operators_by_pivoting(Hx, Hz)

    logical_error_rates = []
    stderrs = []
    runtimes = []

    print(f"BP max iterations: {bp_max_iter}, OSD order: {osd_order}, MS scaling factor: {ms_scaling_factor}")

    for p in p_vals:
        # print(f"Evaluating for p={p}...")
        # if the name of parity check matrix and the logical operators are Hz and Lz, respectively, then it's bit-flip (X) error decoder
        bp_osd_decoder = BpOsdDecoder(
            pcm=Hz,
            error_rate=float(p),
            max_iter=bp_max_iter,
            bp_method='minimum_sum',
            ms_scaling_factor=ms_scaling_factor,
            schedule='parallel',
            osd_method='OSD_CS',
            osd_order=osd_order,
        )

        logical_error_rate, stderr, runtime = compute_logical_error_rate(Hz, Lz, p, run_count=MC_budget, DECODER=bp_osd_decoder, run_label=run_label, DEBUG=False)
        
        logical_error_rates.append(logical_error_rate)
        stderrs.append(stderr)
        runtimes.append(runtime)

    return {
        "logical_error_rates": logical_error_rates,
        "stderrs": stderrs,
        **base_payload,
        "runtimes": runtimes,
        "skipped": False
    }