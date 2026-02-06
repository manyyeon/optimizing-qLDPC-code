from logical_operators import get_logical_operators_by_pivoting
from decoder_performance import compute_logical_error_rate
import ldpc
import ldpc.code_util
from ldpc.bposd_decoder import BpOsdDecoder
from basic_css_code import construct_HGP_code
from optimization.experiments_settings import tanner_graph_to_parity_check_matrix
from optimization.experiments_settings import (
    generate_neighbor_highlight, load_tanner_graph, parse_edgelist,
    codes, path_to_initial_codes, textfiles, noise_levels
)
import time
import numpy as np
import argparse
from tqdm import tqdm
import h5py
import sys
import os
import networkx as nx
from scipy.sparse import csr_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- IMPORTS FROM YOUR PROJECT ---

# --- CONFIGURATION ---
exploration_params = [(24, 40), (15, 70), (12, 40), (12, 40)]  # (N, L)
output_file = "optimization/results/batch_search_d_first.hdf5"

# MC Budgets
BUDGET_SCREENING = 10_000   # Phase 2
BUDGET_PRECISION = 100_000  # Phase 3
TOP_K_PRECISION = 10        # How many to promote to Phase 3

# ----------------- HELPER FUNCTIONS -----------------


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
            csr_matrix(H.T, dtype=np.uint8))

    # Construct Quantum Codes
    Hx, Hz = construct_HGP_code(H)

    n_q = n_cl**2 + n_T_cl**2
    k_q = k_cl**2 + k_T_cl**2
    d_q = min(d_cl, d_T_cl)  # Lower bound estimate

    params = {
        "n_classical": n_cl, "k_classical": k_cl, "d_classical": d_cl,
        "n_T_classical": n_T_cl, "k_T_classical": k_T_cl, "d_T_classical": d_T_cl,
        "rank_H": r_cl,
        "n_quantum": n_q, "k_quantum": k_q, "d_quantum": d_q,
    }
    return params, Hx, Hz


def evaluate_mc(Hx, Hz, p, budget, run_label="eval"):
    """Runs Monte Carlo simulation only."""
    if budget == 0:
        return 0.0, 0.0, 0.0

    bp_max_iter = int(Hx.shape[1]/10)
    Lx, Lz = get_logical_operators_by_pivoting(Hx, Hz)

    bp_osd_decoder = BpOsdDecoder(
        pcm=Hz, error_rate=float(p), max_iter=bp_max_iter,
        bp_method='minimum_sum', ms_scaling_factor=0.625,
        schedule='parallel', osd_method='OSD_CS', osd_order=2,
    )

    ler, stderr, runtime = compute_logical_error_rate(
        Hz, Lz, p, run_count=budget, DECODER=bp_osd_decoder,
        run_label=run_label, DEBUG=False
    )
    return ler, stderr, runtime

# --- HDF5 UTILS ---


def _ensure_ds(grp, name, shape_sample, dtype):
    if name in grp:
        return grp[name]
    shape = (0,) + shape_sample
    maxshape = (None,) + shape_sample
    return grp.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True)


def append_to_hdf5(grp, edge_list, params, ler=0.0, std=0.0, runtime=0.0):
    # Ensure datasets exist
    ds_states = _ensure_ds(grp, "states", (edge_list.shape[0],), np.uint32)
    ds_ler = _ensure_ds(grp, "logical_error_rates", (), np.float64)
    ds_std = _ensure_ds(grp, "logical_error_rates_std", (), np.float64)
    ds_dcl = _ensure_ds(grp, "distances_classical", (), np.float64)
    ds_dclT = _ensure_ds(grp, "distances_classical_T", (), np.float64)
    ds_dq = _ensure_ds(grp, "distances_quantum", (), np.float64)
    ds_run = _ensure_ds(grp, "decoding_runtimes", (), np.float64)

    # Append
    idx = ds_ler.shape[0]
    ds_states.resize(idx+1, axis=0)
    ds_states[idx] = edge_list
    ds_ler.resize(idx+1, axis=0)
    ds_ler[idx] = ler
    ds_std.resize(idx+1, axis=0)
    ds_std[idx] = std
    ds_dcl.resize(idx+1, axis=0)
    ds_dcl[idx] = params['d_classical']
    ds_dclT.resize(idx+1, axis=0)
    ds_dclT[idx] = params['d_T_classical']
    ds_dq.resize(idx+1, axis=0)
    ds_dq[idx] = params['d_quantum']
    ds_run.resize(idx+1, axis=0)
    ds_run[idx] = runtime

    return idx


def update_hdf5_row(grp, idx, ler, std, runtime):
    grp["logical_error_rates"][idx] = ler
    grp["logical_error_rates_std"][idx] = std
    grp["decoding_runtimes"][idx] += runtime  # Add to existing (if any)

# ----------------- MAIN -----------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', default=0, type=int)
    parser.add_argument('-N', default=None, type=int)
    parser.add_argument('-L', default=None, type=int)
    parser.add_argument('-p', default=None, type=float)
    args = parser.parse_args()

    C = args.C
    N, L = exploration_params[C] if (args.N is None) else (args.N, args.L)
    p = noise_levels[C] if args.p is None else args.p

    print(f"--- BATCH OPTIMIZATION ---")
    print(f"Code Family: {codes[C]}, p={p}")
    print(f"Search: {L} steps x {N} neighbors = {L*N} total scans")
    print(
        f"Budgets: Screening={BUDGET_SCREENING}, Precision={BUDGET_PRECISION}")

    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    with h5py.File(output_file, "a") as f:
        grp = f.require_group(codes[C])
        grp.attrs.update(
            {'N': N, 'L': L, 'p': p, 'budget_screen': BUDGET_SCREENING, 'budget_prec': BUDGET_PRECISION})

        # --- PHASE 1: GEOMETRIC SEARCH (RANDOM WALK) ---
        print("\n>>> PHASE 1: Geometric Search (Distance Only)")

        current_state = initial_state
        # Candidates list: stores dicts of {'state', 'params', 'row_idx', 'dist'}
        # We start with the initial state
        params_init, _, _ = get_code_parameters_and_matrices(initial_state)
        edges_init = parse_edgelist(initial_state).astype(np.uint32)
        idx_init = append_to_hdf5(grp, edges_init, params_init)

        all_candidates = []  # Store ALL valid candidates here
        current_dist = min(
            params_init['d_classical'], params_init['d_T_classical'])

        # Add initial state to candidates
        all_candidates.append(
            {'state': initial_state, 'params': params_init, 'row_idx': idx_init, 'dist': current_dist})

        global_max_dist = current_dist

        for step in range(L):
            print(f"Step {step+1}/{L} (Current Max Dist: {global_max_dist})")

            step_best_dist = -1
            step_best_neighbor = None

            for _ in tqdm(range(N), desc="Scanning Neighbors", leave=False):
                neighbor, _, _ = generate_neighbor_highlight(current_state)
                params, _, _ = get_code_parameters_and_matrices(neighbor)

                # Calculate distance
                d_cand = int(min(params['d_classical'],
                             params['d_T_classical']))

                # Save to disk (LER=0 for now)
                edges = parse_edgelist(neighbor).astype(np.uint32)
                row_idx = append_to_hdf5(grp, edges, params)

                # Track candidate
                cand_entry = {'state': neighbor, 'params': params,
                              'row_idx': row_idx, 'dist': d_cand}
                all_candidates.append(cand_entry)

                # Update Max Distance found globally
                if d_cand > global_max_dist:
                    global_max_dist = d_cand
                    # print(f"  New Global Max Distance found: {global_max_dist}")

                # Logic for Next Step in Walk: Steepest Ascent on Distance
                if d_cand > step_best_dist:
                    step_best_dist = d_cand
                    step_best_neighbor = neighbor
                elif d_cand == step_best_dist:
                    # Tie-breaker: random or keep current (here we just update to latest)
                    step_best_neighbor = neighbor

            # Move walker
            if step_best_neighbor is not None:
                current_state = step_best_neighbor

            f.flush()

        # --- PHASE 2: SCREENING (10^4 MC) ---
        print(f"\n>>> PHASE 2: Screening (Budget {BUDGET_SCREENING})")

        # Filter: Only evaluate codes that achieved the GLOBAL MAX DISTANCE
        # (You could also lower this to global_max_dist - 2 if you want more diversity)
        target_dist = global_max_dist
        screening_candidates = [
            c for c in all_candidates if c['dist'] == target_dist]

        # Deduplicate based on row_idx to avoid re-running same code if found multiple times
        # (Though usually unique edges implies unique row, list might contain duplicates if visited twice)
        # Using a dict by row_idx to dedup
        unique_candidates = {c['row_idx']                             : c for c in screening_candidates}.values()

        print(f"Found {len(all_candidates)} total codes.")
        print(
            f"Screening {len(unique_candidates)} unique candidates with Dist == {target_dist}...")

        screened_results = []

        for cand in tqdm(unique_candidates, desc="Running MC (Screening)"):
            params, Hx, Hz = get_code_parameters_and_matrices(cand['state'])
            ler, std, run_t = evaluate_mc(Hx, Hz, p, BUDGET_SCREENING)

            # Update HDF5
            update_hdf5_row(grp, cand['row_idx'], ler, std, run_t)

            cand['ler'] = ler
            screened_results.append(cand)
            f.flush()

        if not screened_results:
            print("No candidates to screen.")
            sys.exit()

        # Sort by LER (ascending)
        screened_results.sort(key=lambda x: x['ler'])
        print(f"Best Screening LER: {screened_results[0]['ler']:.2e}")

        # --- PHASE 3: PRECISION (10^5 MC) ---
        print(f"\n>>> PHASE 3: Precision (Budget {BUDGET_PRECISION})")

        top_candidates = screened_results[:TOP_K_PRECISION]
        print(f"Promoting Top {len(top_candidates)} to high precision...")

        final_best_cand = None
        min_final_ler = np.inf

        for cand in top_candidates:
            print(f"Evaluating Candidate (Row {cand['row_idx']})...")
            params, Hx, Hz = get_code_parameters_and_matrices(cand['state'])

            # Note: We run FRESH 10^5, or you could add 90k to the existing 10k.
            # Here we run fresh 100k for clean stats, but you overwrite the HDF5 row.
            ler, std, run_t = evaluate_mc(Hx, Hz, p, BUDGET_PRECISION)

            # Overwrite with high-precision result
            update_hdf5_row(grp, cand['row_idx'], ler, std, run_t)

            print(f"  -> LER: {ler:.2e}")

            if ler < min_final_ler:
                min_final_ler = ler
                final_best_cand = cand

            f.flush()

        # Save Best State Metadata
        if final_best_cand:
            print(f"\n>>> RESULT: Best Code Found")
            print(f"Distance: {final_best_cand['dist']}")
            print(f"LER: {min_final_ler:.2e}")

            best_edges = parse_edgelist(
                final_best_cand['state']).astype(np.uint32)
            if "best_state" in grp:
                del grp["best_state"]
            grp.create_dataset(
                "best_state", data=best_edges[np.newaxis, :], dtype=np.uint32)
            grp.attrs['min_cost'] = min_final_ler

    print(f"Saved to {output_file}")
