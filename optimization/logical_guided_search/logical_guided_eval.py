import numpy as np
import networkx as nx
import ldpc
import ldpc.code_util
from ldpc.bposd_decoder import BpOsdDecoder
from scipy.sparse import csr_matrix

from decoder_performance import (
    compute_logical_error_rate,
    compute_logical_error_rate_parallel_batched,
    compute_logical_error_rate_depolarizing,
    compute_logical_error_rate_depolarizing_parallel_batched,
)
from logical_operators import get_logical_operators_by_pivoting
from basic_css_code import construct_HGP_code
from optimization.experiments_settings import tanner_graph_to_parity_check_matrix

from optimization.analyze_codes.count_low_weight_patterns import (
    count_parent_low_weight_patterns,
)


def _get_count(counts_total, w: int) -> int:
    """
    Robustly read count for weight w from either dict-like or array-like counts.
    """
    if isinstance(counts_total, dict):
        return int(counts_total.get(w, counts_total.get(str(w), 0)))

    if w < len(counts_total):
        return int(counts_total[w])

    return 0


def compute_weighted_low_weight_score(
    state: nx.MultiGraph,
    params: dict | None = None,
    max_weight_offset: int = 2,
    score_mode: str = "absolute",
    gamma: float = 0.1,
    max_weight: int | None = None,
):
    """
    Compute low-weight parent-code spectrum score.

    Modes
    -----
    relative:
        S_{W,gamma}
          = sum_{w=d_Q}^{d_Q+offset}
              gamma^{w-d_Q} A_w

        This should be used with distance-first ranking.

    absolute:
        S_abs
          = sum_{w=1}^{Wmax}
              gamma^{w-Wmax} A_w

        This can be used as a single score across different distances.

    Lower score is better in both modes.
    """
    if params is None:
        params, _, _ = get_code_parameters_and_matrices(state)

    d_q = int(min(params["d_classical"], params["d_T_classical"]))

    if score_mode not in {"relative", "absolute"}:
        raise ValueError(f"Unknown score_mode={score_mode!r}")

    if score_mode == "relative":
        min_weight = d_q
        max_weight_used = d_q + max_weight_offset
        weight_base = gamma
    else:
        if max_weight is None:
            raise ValueError(
                "For score_mode='absolute', max_weight must be provided.")
        if not (0.0 < gamma < 1.0):
            raise ValueError("gamma must satisfy 0 < gamma < 1.")
        min_weight = 1
        max_weight_used = int(max_weight)
        weight_base = gamma

    H = tanner_graph_to_parity_check_matrix(state)
    counts = count_parent_low_weight_patterns(H, max_weight=max_weight_used)
    counts_total = counts["counts_total"]

    score = 0.0
    components = {}
    weights = {}

    for w in range(min_weight, max_weight_used + 1):
        count_w = _get_count(counts_total, w)
        components[w] = count_w

        if score_mode == "relative":
            coeff = gamma ** (w - d_q)
        else:
            coeff = gamma ** (w - max_weight_used)

        weights[w] = float(coeff)
        score += float(count_w) * float(coeff)

    return {
        "score": float(score),
        "score_mode": score_mode,
        "d_q": d_q,
        "min_weight": min_weight,
        "max_weight": max_weight_used,
        "gamma": gamma,
        "components": components,
        "weights": weights,
    }


def get_code_parameters_and_matrices(state: nx.MultiGraph):
    H = tanner_graph_to_parity_check_matrix(state)
    csr_H = csr_matrix(H, dtype=np.uint8)

    n_cl, k_cl, _ = ldpc.code_util.compute_code_parameters(csr_H)
    d_cl = ldpc.code_util.compute_exact_code_distance(csr_H)
    r_cl = ldpc.mod2.rank(csr_H)

    m_cl = csr_H.shape[0]

    if r_cl == m_cl:
        n_T_cl = m_cl
        k_T_cl = 0
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


def evaluate_mc(
    Hx,
    Hz,
    p,
    budget,
    run_label="eval",
    failure_cap=None,
    min_runs_before_stop=0,
    workers=1,
    batch_size=10000,
):
    if budget == 0:
        return {
            "ler": 0.0,
            "stderr": 0.0,
            "runtime": 0.0,
            "failures": 0,
            "completed_runs": 0,
            "early_stopped": False,
        }

    _, Lz = get_logical_operators_by_pivoting(Hx, Hz)

    if workers == 1:
        bp_max_iter = int(Hx.shape[1] / 10)

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

        ler, stderr, runtime, failures, completed_runs, early_stopped = compute_logical_error_rate(
            H=Hz,
            L=Lz,
            error_rate=p,
            run_count=budget,
            DECODER=bp_osd_decoder,
            run_label=run_label,
            DEBUG=False,
            failure_cap=failure_cap,
            min_runs_before_stop=min_runs_before_stop,
        )

    else:
        ler, stderr, runtime, failures, completed_runs, early_stopped = compute_logical_error_rate_parallel_batched(
            Hz=Hz,
            Lz=Lz,
            error_rate=p,
            run_count=budget,
            run_label=run_label,
            workers=workers,
            batch_size=batch_size,
            failure_cap=failure_cap,
            min_runs_before_stop=min_runs_before_stop,
        )

    return {
        "ler": ler,
        "stderr": stderr,
        "runtime": runtime,
        "failures": failures,
        "completed_runs": completed_runs,
        "early_stopped": early_stopped,
    }

def evaluate_mc_depolarizing(
    Hx,
    Hz,
    p,
    budget,
    run_label="depolarizing_eval",
    failure_cap=None,
    min_runs_before_stop=0,
    workers=1,
    batch_size=50000,
):
    """
    Evaluate a CSS code under code-capacity depolarizing noise.

    Here p is the total physical Pauli error probability:
        P(X) = P(Y) = P(Z) = p/3.
    """
    if budget == 0:
        return {
            "ler": 0.0,
            "stderr": 0.0,
            "runtime": 0.0,
            "failures": 0,
            "completed_runs": 0,
            "early_stopped": False,
        }

    Lx, Lz = get_logical_operators_by_pivoting(Hx, Hz)

    if workers == 1:
        (
            ler,
            stderr,
            runtime,
            failures,
            completed_runs,
            early_stopped,
        ) = compute_logical_error_rate_depolarizing(
            Hx=Hx,
            Hz=Hz,
            Lx=Lx,
            Lz=Lz,
            error_rate=p,
            run_count=budget,
            run_label=run_label,
            failure_cap=failure_cap,
            min_runs_before_stop=min_runs_before_stop,
        )
    else:
        (
            ler,
            stderr,
            runtime,
            failures,
            completed_runs,
            early_stopped,
        ) = compute_logical_error_rate_depolarizing_parallel_batched(
            Hx=Hx,
            Hz=Hz,
            Lx=Lx,
            Lz=Lz,
            error_rate=p,
            run_count=budget,
            run_label=run_label,
            workers=workers,
            batch_size=batch_size,
            failure_cap=failure_cap,
            min_runs_before_stop=min_runs_before_stop,
        )

    return {
        "ler": ler,
        "stderr": stderr,
        "runtime": runtime,
        "failures": failures,
        "completed_runs": completed_runs,
        "early_stopped": early_stopped,
    }