import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

import networkx as nx
from scipy.sparse import csr_matrix



# --- IMPORTS FROM YOUR PROJECT ---

# --- CONFIGURATION ---
exploration_params = [(50, 10), (50, 10), (50, 10), (50, 10)]  # (N, L)
output_file = os.environ.get('OUTPUT_HDF5', "optimization/results/best_neighbor_search_d_first_2.hdf5")

# MC Budgets
BUDGET_SCREENING = 10_000   # Phase 2
BUDGET_PRECISION = 100_000  # Phase 3
TOP_K_PRECISION = 3        # How many to promote to Phase 3
SIGNATURE_PRINT_LIMIT = 30  # Max number of signature examples to print

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
    reg_metrics = compute_regularity_metrics(H)
    return params, Hx, Hz, reg_metrics


def compute_regularity_metrics(H: np.ndarray) -> dict:
    row_w = np.asarray(H.sum(axis=1)).ravel().astype(int)
    col_w = np.asarray(H.sum(axis=0)).ravel().astype(int)

    def _entropy(counts):
        counts = np.asarray(counts, dtype=np.float64)
        total = counts.sum()
        if total <= 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return float(-(probs * np.log2(probs)).sum())

    row_var = float(np.var(row_w)) if row_w.size else 0.0
    col_var = float(np.var(col_w)) if col_w.size else 0.0
    row_entropy = _entropy(np.bincount(row_w)) if row_w.size else 0.0
    col_entropy = _entropy(np.bincount(col_w)) if col_w.size else 0.0

    def _canonical_shift(sig, modulo):
        reps = []
        for s in range(modulo):
            reps.append(tuple(sorted(((i + s) % modulo) for i in sig)))
        return min(reps)

    signatures = set()
    for j in range(H.shape[1]):
        sig = tuple(np.flatnonzero(H[:, j]).tolist())
        signatures.add(sig)

    # Cyclic-class counts (check patterns: rows shifted by columns; variable patterns: columns shifted by rows)
    check_patterns = [tuple(np.flatnonzero(H[i, :]).tolist()) for i in range(H.shape[0])]
    var_patterns = [tuple(np.flatnonzero(H[:, j]).tolist()) for j in range(H.shape[1])]
    check_cyclic_classes = len({_canonical_shift(sig, H.shape[1]) for sig in check_patterns})
    var_cyclic_classes = len({_canonical_shift(sig, H.shape[0]) for sig in var_patterns})

    signature_examples = sorted(signatures)[:SIGNATURE_PRINT_LIMIT]

    return {
        "row_weights": row_w,
        "col_weights": col_w,
        "row_weight_var": row_var,
        "col_weight_var": col_var,
        "distinct_col_signatures": int(len(signatures)),
        "signature_examples": signature_examples,
        "check_cyclic_classes": check_cyclic_classes,
        "var_cyclic_classes": var_cyclic_classes,
        "row_degree_entropy": row_entropy,
        "col_degree_entropy": col_entropy,
    }


def _format_reg_metrics(metrics: dict) -> str:
    return (
        f"row_weights={metrics['row_weights']}, col_weights={metrics['col_weights']}, "
        f"row_var={metrics['row_weight_var']:.3f}, col_var={metrics['col_weight_var']:.3f}, "
        f"sig_cnt={metrics['distinct_col_signatures']}, "
        f"check_cyclic_classes={metrics['check_cyclic_classes']}, "
        f"var_cyclic_classes={metrics['var_cyclic_classes']}, "
        f"row_ent={metrics['row_degree_entropy']:.3f}, col_ent={metrics['col_degree_entropy']:.3f}, "
        f"signature_examples={metrics['signature_examples']}"
    )


def _format_signature_examples(metrics: dict) -> str:
    examples = metrics.get("signature_examples", [])
    return f"signature_examples={examples}"


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

    res = compute_logical_error_rate(
        Hz, Lz, p, run_count=budget, DECODER=bp_osd_decoder,
        run_label=run_label, DEBUG=False
    )

    # compute_logical_error_rate may return multiple diagnostics; take first three
    if isinstance(res, (list, tuple)) and len(res) >= 3:
        ler, stderr, runtime = res[0], res[1], res[2]
    else:
        # Fallback
        ler, stderr, runtime = res, 0.0, 0.0

    return ler, stderr, runtime

# --- HDF5 UTILS ---


def _ensure_ds(grp, name, shape_sample, dtype):
    if name in grp:
        return grp[name]
    shape = (0,) + shape_sample
    maxshape = (None,) + shape_sample
    return grp.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True)


def append_to_hdf5(grp, edge_list, params, reg_metrics, ler=0.0, std=0.0, runtime=0.0, wall_time=np.nan):
    # Ensure datasets exist
    ds_states = _ensure_ds(grp, "states", (edge_list.shape[0],), np.uint32)
    ds_ler = _ensure_ds(grp, "logical_error_rates", (), np.float64)
    ds_std = _ensure_ds(grp, "logical_error_rates_std", (), np.float64)
    ds_dcl = _ensure_ds(grp, "distances_classical", (), np.float64)
    ds_dclT = _ensure_ds(grp, "distances_classical_T", (), np.float64)
    ds_dq = _ensure_ds(grp, "distances_quantum", (), np.float64)
    ds_run = _ensure_ds(grp, "decoding_runtimes", (), np.float64)
    ds_wall = _ensure_ds(grp, "wall_times", (), np.float64)
    ds_ler_wall = _ensure_ds(grp, "ler_wall_times", (), np.float64)
    ds_row_var = _ensure_ds(grp, "row_weight_var", (), np.float64)
    ds_col_var = _ensure_ds(grp, "col_weight_var", (), np.float64)
    ds_sig_cnt = _ensure_ds(grp, "distinct_col_signatures", (), np.int32)
    ds_row_ent = _ensure_ds(grp, "row_degree_entropy", (), np.float64)
    ds_col_ent = _ensure_ds(grp, "col_degree_entropy", (), np.float64)
    ds_chk_cyc = _ensure_ds(grp, "check_cyclic_classes", (), np.int32)
    ds_var_cyc = _ensure_ds(grp, "var_cyclic_classes", (), np.int32)

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
    ds_wall.resize(idx+1, axis=0)
    ds_wall[idx] = wall_time
    ds_ler_wall.resize(idx+1, axis=0)
    ds_ler_wall[idx] = wall_time if ler > 0 else np.nan

    ds_row_var.resize(idx+1, axis=0)
    ds_row_var[idx] = reg_metrics['row_weight_var']
    ds_col_var.resize(idx+1, axis=0)
    ds_col_var[idx] = reg_metrics['col_weight_var']
    ds_sig_cnt.resize(idx+1, axis=0)
    ds_sig_cnt[idx] = reg_metrics['distinct_col_signatures']
    ds_row_ent.resize(idx+1, axis=0)
    ds_row_ent[idx] = reg_metrics['row_degree_entropy']
    ds_col_ent.resize(idx+1, axis=0)
    ds_col_ent[idx] = reg_metrics['col_degree_entropy']
    ds_chk_cyc.resize(idx+1, axis=0)
    ds_chk_cyc[idx] = reg_metrics['check_cyclic_classes']
    ds_var_cyc.resize(idx+1, axis=0)
    ds_var_cyc[idx] = reg_metrics['var_cyclic_classes']

    return idx


def update_hdf5_row(grp, idx, ler, std, runtime, wall_time=np.nan):
    grp["logical_error_rates"][idx] = ler
    grp["logical_error_rates_std"][idx] = std
    grp["decoding_runtimes"][idx] += runtime  # Add to existing (if any)
    if "ler_wall_times" in grp:
        grp["ler_wall_times"][idx] = wall_time

# ----------------- MAIN -----------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', default=0, type=int)
    parser.add_argument('-N', default=None, type=int)
    parser.add_argument('-L', default=None, type=int)
    parser.add_argument('-p', default=None, type=float)
    parser.add_argument('--patience', default=5, type=int,
                        help='Early-stop patience in Phase 1 (steps without improvement)')
    parser.add_argument('--target-distance', default=None, type=int,
                        help='Stop early when distance >= target (min of d_classical, d_T_classical).')
    parser.add_argument('--stop-at-target', action='store_true',
                        help='Stop the run immediately after reaching target distance (skip LER phases).')
    parser.add_argument('--budget-screening', default=BUDGET_SCREENING, type=int,
                        help='MC budget for screening phase (Phase 2).')
    parser.add_argument('--budget-precision', default=BUDGET_PRECISION, type=int,
                        help='MC budget for precision phase (Phase 3).')
    parser.add_argument('--top-k-precision', default=TOP_K_PRECISION, type=int,
                        help='Number of candidates promoted to precision phase.')
    parser.add_argument('--screen-top-k', default=None, type=int,
                        help='Screen only the top-k candidates by distance (Phase 2).')
    parser.add_argument('--skip-screening', action='store_true',
                        help='Skip screening and precision phases (Phase 2/3).')
    parser.add_argument('--skip-precision', action='store_true',
                        help='Skip precision phase (Phase 3).')
    parser.add_argument('--print-patterns', action='store_true',
                        help='Print sample signature patterns when distance improves.')
    args = parser.parse_args()

    C = args.C
    N, L = exploration_params[C] if (args.N is None) else (args.N, args.L)
    p = noise_levels[C] if args.p is None else args.p
    target_distance = args.target_distance
    stop_at_target = args.stop_at_target
    budget_screening = args.budget_screening
    budget_precision = args.budget_precision
    top_k_precision = args.top_k_precision
    screen_top_k = args.screen_top_k
    print_patterns = args.print_patterns
    skip_screening = args.skip_screening or budget_screening <= 0
    skip_precision = args.skip_precision or budget_precision <= 0 or top_k_precision <= 0

    print(f"--- BATCH OPTIMIZATION ---")
    print(f"Code Family: {codes[C]}, p={p}")
    print(f"Search: {L} steps x {N} neighbors = {L*N} total scans")
    print(
        f"Budgets: Screening={budget_screening}, Precision={budget_precision}, TopK={top_k_precision}")

    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    start_time = time.time()

    with h5py.File(output_file, "a") as f:
        grp = f.require_group(codes[C])
        grp.attrs.update(
            {
                'N': N,
                'L': L,
                'p': p,
                'budget_screen': budget_screening,
                'budget_prec': budget_precision,
                'top_k_precision': top_k_precision,
                'screen_top_k': -1 if screen_top_k is None else screen_top_k,
                'target_distance': -1 if target_distance is None else target_distance,
                'stop_at_target': bool(stop_at_target),
                'skip_screening': bool(skip_screening),
                'skip_precision': bool(skip_precision),
            })

        # --- PHASE 1: GEOMETRIC SEARCH (RANDOM WALK) ---
        print("\n>>> PHASE 1: Geometric Search (Distance Only)")

        current_state = initial_state
        # Candidates list: stores dicts of {'state', 'params', 'row_idx', 'dist'}
        # We start with the initial state
        params_init, _, _, reg_init = get_code_parameters_and_matrices(initial_state)
        edges_init = parse_edgelist(initial_state).astype(np.uint32)
        idx_init = append_to_hdf5(
            grp, edges_init, params_init, reg_init, wall_time=time.time() - start_time)

        all_candidates = []  # Store ALL valid candidates here
        current_dist = min(
            params_init['d_classical'], params_init['d_T_classical'])

        print(f"Initial distance: {current_dist}. Regularity: {_format_reg_metrics(reg_init)}")
        if print_patterns:
            print(f"  {_format_signature_examples(reg_init)}")

        # Add initial state to candidates
        all_candidates.append(
            {'state': initial_state, 'params': params_init, 'row_idx': idx_init, 'dist': current_dist})

        # Track seen states (to avoid re-evaluating identical candidates)
        seen_keys = set()
        try:
            seen_keys.add(tuple(parse_edgelist(initial_state).tolist()))
        except Exception:
            pass

        global_max_dist = current_dist
        target_reached = False
        stop_after_phase1 = False

        if target_distance is not None and current_dist >= target_distance:
            grp.attrs['time_to_target'] = 0.0
            grp.attrs['target_reached'] = True
            target_reached = True
            if stop_at_target:
                print(f"Target distance {target_distance} reached at initial state. Stopping early.")
                stop_after_phase1 = True

        # Early-stop patience: stop if no improvement in `patience` steps
        patience = args.patience
        steps_since_improve = 0
        prev_global_max = global_max_dist

        for step in range(L):
            print(f"Step {step+1}/{L} (Current Max Dist: {global_max_dist})")

            step_best_dist = -1
            step_best_neighbor = None

            for _ in tqdm(range(N), desc="Scanning Neighbors", leave=False):
                neighbor, _, _ = generate_neighbor_highlight(current_state)
                # deduplicate by exact edge list if we've seen this configuration before
                try:
                    key = tuple(parse_edgelist(neighbor).tolist())
                except Exception:
                    key = None

                if key is not None and key in seen_keys:
                    continue

                params, _, _, reg_metrics = get_code_parameters_and_matrices(neighbor)

                # Calculate distance
                d_cand = int(min(params['d_classical'],
                             params['d_T_classical']))

                # Save to disk (LER=0 for now)
                edges = parse_edgelist(neighbor).astype(np.uint32)
                row_idx = append_to_hdf5(
                    grp, edges, params, reg_metrics, wall_time=time.time() - start_time)

                # Track candidate
                cand_entry = {'state': neighbor, 'params': params,
                              'row_idx': row_idx, 'dist': d_cand, 'reg_metrics': reg_metrics}
                all_candidates.append(cand_entry)

                if key is not None:
                    seen_keys.add(key)

                # Update Max Distance found globally
                if d_cand > global_max_dist:
                    global_max_dist = d_cand
                    print(f"  New max dist {global_max_dist}. Regularity: {_format_reg_metrics(reg_metrics)}")
                    if print_patterns:
                        print(f"  {_format_signature_examples(reg_metrics)}")
                    # print(f"  New Global Max Distance found: {global_max_dist}")

                if (not target_reached) and target_distance is not None and d_cand >= target_distance:
                    grp.attrs['time_to_target'] = time.time() - start_time
                    grp.attrs['target_reached'] = True
                    target_reached = True
                    if stop_at_target:
                        print(f"Target distance {target_distance} reached. Stopping after Phase 1.")
                        stop_after_phase1 = True
                        break

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

            grp.attrs['total_runtime'] = time.time() - start_time
            f.flush()

            # Check for early stopping on distance convergence
            if global_max_dist > prev_global_max:
                steps_since_improve = 0
                prev_global_max = global_max_dist
            else:
                steps_since_improve += 1

            if steps_since_improve >= patience:
                print(f"Stopping early after {steps_since_improve} steps with no improvement.")
                break

            if stop_after_phase1:
                break

        if stop_after_phase1 or skip_screening:
            if stop_after_phase1:
                grp.attrs['stopped_at_target'] = True
            if skip_screening:
                print("Skipping screening/precision phases by request.")
            f.flush()
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total Time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s ({total_time:.2f}s)")
            print(f"Saved to {output_file}")
            sys.exit()

        # --- PHASE 2: SCREENING (10^4 MC) ---
        print(f"\n>>> PHASE 2: Screening (Budget {budget_screening})")

        # Candidate selection for screening
        if screen_top_k is not None and screen_top_k > 0:
            # Deduplicate based on row_idx to avoid re-running same code if found multiple times
            unique_candidates = list({c['row_idx']: c for c in all_candidates}.values())
            unique_candidates.sort(key=lambda x: x['dist'], reverse=True)
            unique_candidates = unique_candidates[:screen_top_k]
            if unique_candidates:
                best_dist = unique_candidates[0]['dist']
                worst_dist = unique_candidates[-1]['dist']
            else:
                best_dist = None
                worst_dist = None
            print(f"Found {len(all_candidates)} total codes.")
            print(f"Screening top {len(unique_candidates)} candidates by distance (best={best_dist}, worst={worst_dist})...")
        else:
            # Filter: Only evaluate codes that achieved the GLOBAL MAX DISTANCE
            target_dist = global_max_dist
            screening_candidates = [c for c in all_candidates if c['dist'] == target_dist]

            # Deduplicate based on row_idx to avoid re-running same code if found multiple times
            unique_candidates = {c['row_idx']: c for c in screening_candidates}.values()

            print(f"Found {len(all_candidates)} total codes.")
            print(
                f"Screening {len(unique_candidates)} unique candidates with Dist == {target_dist}...")

        screened_results = []

        for cand in tqdm(unique_candidates, desc="Running MC (Screening)"):
            params, Hx, Hz, _ = get_code_parameters_and_matrices(cand['state'])
            ler, std, run_t = evaluate_mc(Hx, Hz, p, budget_screening)

            # Update HDF5
            update_hdf5_row(
                grp, cand['row_idx'], ler, std, run_t, wall_time=time.time() - start_time)

            cand['ler'] = ler
            screened_results.append(cand)
            f.flush()

        if not screened_results:
            print("No candidates to screen.")
            sys.exit()

        # Sort by LER (ascending)
        screened_results.sort(key=lambda x: x['ler'])
        print(f"Best Screening LER: {screened_results[0]['ler']:.4f} (Dist: {screened_results[0]['dist']})")

        # --- PHASE 3: PRECISION (10^5 MC) ---
        if skip_precision:
            print("Skipping precision phase by request.")
            best_cand = screened_results[0]
            best_edges = parse_edgelist(best_cand['state']).astype(np.uint32)
            if "best_state" in grp:
                del grp["best_state"]
            grp.create_dataset(
                "best_state", data=best_edges[np.newaxis, :], dtype=np.uint32)
            grp.attrs['min_cost'] = float(best_cand['ler'])
            grp.attrs['total_runtime'] = time.time() - start_time
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total Time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s ({total_time:.2f}s)")
            print(f"Saved to {output_file}")
            sys.exit()

        print(f"\n>>> PHASE 3: Precision (Budget {budget_precision})")

        top_candidates = screened_results[:top_k_precision]
        print(f"Promoting Top {len(top_candidates)} to high precision...")
        print("Candidates:")
        for i, c in enumerate(top_candidates):
            print(
                f"  {i+1}. Row {c['row_idx']} - Dist: {c['dist']}, LER: {c['ler']:.4f}")

        final_best_cand = None
        min_final_ler = np.inf

        for cand in top_candidates:
            print(f"Evaluating Candidate (Row {cand['row_idx']} - LER {cand['ler']:.4f})...")
            params, Hx, Hz, _ = get_code_parameters_and_matrices(cand['state'])

            # Note: We run FRESH 10^5, or you could add 90k to the existing 10k.
            # Here we run fresh 100k for clean stats, but you overwrite the HDF5 row.
            ler, std, run_t = evaluate_mc(Hx, Hz, p, budget_precision)

            # Overwrite with high-precision result
            update_hdf5_row(
                grp, cand['row_idx'], ler, std, run_t, wall_time=time.time() - start_time)

            print(f"  -> LER: {ler:.6f} ± {std:.6f} (Runtime: {run_t:.2f}s)")

            if ler < min_final_ler:
                min_final_ler = ler
                final_best_cand = cand

            f.flush()

        # Save Best State Metadata
        if final_best_cand:
            print(f"\n>>> RESULT: Best Code Found")
            print(f"Distance: {final_best_cand['dist']}")
            print(f"LER: {min_final_ler:.6f} ± {std:.6f}")

            best_edges = parse_edgelist(
                final_best_cand['state']).astype(np.uint32)
            if "best_state" in grp:
                del grp["best_state"]
            grp.create_dataset(
                "best_state", data=best_edges[np.newaxis, :], dtype=np.uint32)
            grp.attrs['min_cost'] = min_final_ler

    end_time = time.time()
    total_time = end_time - start_time
    with h5py.File(output_file, "a") as f:
        f[codes[C]].attrs['total_runtime'] = total_time
    print(f"Total Time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s ({total_time:.2f}s)")
    print(f"Saved to {output_file}")
