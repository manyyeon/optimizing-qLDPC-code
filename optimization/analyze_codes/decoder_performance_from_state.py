import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from basic_css_code import construct_HGP_code
from optimization.experiments_settings import tanner_graph_to_parity_check_matrix
import numpy as np

import networkx as nx

from ldpc.bposd_decoder import BpOsdDecoder
from logical_operators import get_logical_operators_by_pivoting
from decoder_performance import compute_logical_error_rate

def compute_decoding_performance_from_state(state: nx.MultiGraph, p_vals: np.ndarray, MC_budget: int) -> dict:
    """
    Evaluate the decoding performance (logical error rates) of a given state.
    Parameters:
        state (nx.MultiGraph): The Tanner graph representing the code.
        p_vals (np.ndarray): Array of physical error rates to evaluate.
        MC_budget (int): The number of Monte Carlo runs to perform for each error rate.
    Returns:
        logical_error_rates (list): List of logical error rates for each physical error rate.
        stds (list): List of standard deviations of the logical error rates.
        runtimes (list): List of runtimes for each decoding operation.
    """

    H = tanner_graph_to_parity_check_matrix(state)

    # H.data = np.where(np.asarray(H.data) > 0, 1, 0)
    Hx, Hz = construct_HGP_code (H)

    bp_max_iter = int(Hx.shape[1]/10)
    # osd_order = 60
    osd_order = 2
    ms_scaling_factor = 0.625

    Lx, Lz = get_logical_operators_by_pivoting(Hx, Hz)

    logical_error_rates = []
    stds = []
    runtimes = []

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
    
        logical_error_rate, std, runtime = compute_logical_error_rate(Hz, Lz, p, run_count=MC_budget, DECODER=bp_osd_decoder, run_label=f"Random exploration", DEBUG=False)
        print(f"Logical error rate for p={p}: {logical_error_rate}")
        
        logical_error_rates.append(logical_error_rate)
        stds.append(std)
        runtimes.append(runtime)

    return {
        "logical_error_rates": logical_error_rates,
        "stds": stds,
        "runtimes": runtimes
    }