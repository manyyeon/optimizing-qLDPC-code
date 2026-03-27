import numpy as np
import networkx as nx
import ldpc
import ldpc.code_util
from ldpc.bposd_decoder import BpOsdDecoder
from scipy.sparse import csr_matrix

from decoder_performance import compute_logical_error_rate
from logical_operators import get_logical_operators_by_pivoting
from basic_css_code import construct_HGP_code
from optimization.experiments_settings import tanner_graph_to_parity_check_matrix


def get_code_parameters_and_matrices(state: nx.MultiGraph):
    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)

    n_cl, k_cl, _ = ldpc.code_util.compute_code_parameters(csr_H)
    d_cl = ldpc.code_util.compute_exact_code_distance(csr_H)
    r_cl = ldpc.mod2.rank(csr_H)

    if k_cl == n_cl - csr_H.shape[0]:
        n_T_cl = csr_H.shape[0]
        k_T_cl = n_T_cl - r_cl
        d_T_cl = np.inf
    else:
        n_T_cl, k_T_cl, d_T_cl = ldpc.code_util.compute_code_parameters(
            csr_matrix(H.T, dtype=np.uint8)
        )

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
    if budget == 0:
        return 0.0, 0.0, 0.0

    bp_max_iter = int(Hx.shape[1] / 10)
    _, Lz = get_logical_operators_by_pivoting(Hx, Hz)

    bp_osd_decoder = BpOsdDecoder(
        pcm=Hz,
        error_rate=float(p),
        max_iter=bp_max_iter,
        bp_method="minimum_sum",
        ms_scaling_factor=0.625,
        schedule="parallel",
        osd_method="OSD_CS",
        osd_order=2,
    )

    ler, stderr, runtime = compute_logical_error_rate(
        Hz,
        Lz,
        p,
        run_count=budget,
        DECODER=bp_osd_decoder,
        run_label=run_label,
        DEBUG=False,
    )
    return ler, stderr, runtime