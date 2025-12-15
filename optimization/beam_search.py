import time
import numpy as np

import argparse
from tqdm import tqdm
import h5py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.analyze_codes.decoder_performance_from_state import evaluate_performance_of_state
from optimization.experiments_settings import generate_neighbor_highlight, load_tanner_graph, parse_edgelist
from optimization.experiments_settings import codes, path_to_initial_codes, textfiles
from optimization.experiments_settings import MC_budget, noise_levels

# exploration_params = [(24, 120), (15, 70), (12, 40), (8, 30)]
exploration_params = [(24, 40), (15, 70), (12, 40), (12, 40)]

output_file = "optimization/results/beam_search_old_parents_included_1e5_run1.hdf5"
EARLY_VALID_TARGET = 10
run_label = "Best neighbor search"
BEAM_WIDTH = 3  # Number of best states to keep at each step
INCLUDE_PARENTS = True  # Whether to include parents in the selection pool

def _current_stream_len(dsets):
    return dsets["logical_error_rates"].shape[0]

def _ensure_ds(grp, name, sample, is_row=True):
    """Create a resizable dataset if it doesn't exist; return the dataset."""
    if name in grp:
        return grp[name]
    arr = np.asarray(sample)
    if arr.ndim == 0:
        shape = (0,)
        maxshape = (None,)
        dtype = arr.dtype
    elif is_row:
        shape = (0, arr.shape[0])
        maxshape = (None, arr.shape[0])
        dtype = arr.dtype
    else:
        shape = (0,) + arr.shape
        maxshape = (None,) + arr.shape
        dtype = arr.dtype
    return grp.create_dataset(name, shape=shape, maxshape=maxshape,
                              dtype=dtype, chunks=True)

def _append_row(ds, row):
    row = np.asarray(row)
    if ds.ndim == 1:
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = row
    else:
        if ds.shape[1] != row.shape[0]:
            raise ValueError(f"Shape mismatch: ds has width {ds.shape[1]} but row has {row.shape[0]}")
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1, :] = row

def _flush_file(f):
    f.flush()
    try:
        fid = f.id.get_vfd_handle()
        if isinstance(fid, int) and fid >= 0:
            os.fsync(fid)
    except Exception:
        pass

def init_dsets(grp, initial_state, initial_result):
    """Create (or fetch) all datasets and return a dict name->dataset."""
    dsets = {}
    dsets["states"]               = _ensure_ds(grp, "states",               parse_edgelist(initial_state).astype(np.uint32), is_row=True)
    dsets["logical_error_rates"]  = _ensure_ds(grp, "logical_error_rates",  np.array(float(initial_result['logical_error_rates'][0]), dtype=np.float64), is_row=False)
    dsets["logical_error_rates_std"] = _ensure_ds(grp, "logical_error_rates_std", np.array(float(initial_result['stderrs'][0]), dtype=np.float64), is_row=False)

    dsets["n_classical"]          = _ensure_ds(grp, "n_classical",          np.array(int(initial_result['n_classical']), dtype=np.int32), is_row=False)
    dsets["n_classical_T"]        = _ensure_ds(grp, "n_classical_T",        np.array(int(initial_result['n_T_classical']), dtype=np.int32), is_row=False)
    dsets["k_classical"]          = _ensure_ds(grp, "k_classical",          np.array(int(initial_result['k_classical']), dtype=np.int32), is_row=False)
    dsets["k_classical_T"]        = _ensure_ds(grp, "k_classical_T",        np.array(int(initial_result['k_T_classical']), dtype=np.int32), is_row=False)
    dsets["rank_H"]               = _ensure_ds(grp, "rank_H",               np.array(int(initial_result['rank_H']), dtype=np.int32), is_row=False)
    dsets["skipped"]              = _ensure_ds(grp, "skipped",              np.array(int(initial_result['skipped']), dtype=np.int32), is_row=False)

    dsets["distances_classical"]   = _ensure_ds(grp, "distances_classical",   np.array(float(initial_result['d_classical']), dtype=np.float64), is_row=False)
    dsets["distances_classical_T"] = _ensure_ds(grp, "distances_classical_T", np.array(float(initial_result['d_T_classical']), dtype=np.float64), is_row=False)
    dsets["distances_quantum"]     = _ensure_ds(grp, "distances_quantum",     np.array(float(initial_result['d_quantum']), dtype=np.float64), is_row=False)
    dsets["distances_Hx"]          = _ensure_ds(grp, "distances_Hx",          np.array(float(initial_result['d_Hx']), dtype=np.float64), is_row=False)
    dsets["distances_Hz"]          = _ensure_ds(grp, "distances_Hz",          np.array(float(initial_result['d_Hz']), dtype=np.float64), is_row=False)
    dsets["decoding_runtimes"]     = _ensure_ds(grp, "decoding_runtimes",     np.array(float(initial_result['runtimes'][0]), dtype=np.float64), is_row=False)
    dsets["step_summaries"]        = _ensure_ds(grp, "step_summaries",        np.array([0, 0, 0, -1], dtype=np.int64), is_row=True)

    # --- NEW: BEAM SURVIVORS DATASET ---
    # Stores the indices of the 'best K' chosen at each step
    # We initialize with -1s. Dimensions: [steps, BEAM_WIDTH]
    sample_beam_row = np.full(BEAM_WIDTH, -1, dtype=np.int64)
    dsets["beam_survivors"] = _ensure_ds(grp, "beam_survivors", sample_beam_row, is_row=True)

    dsets["parent_idx"] = _ensure_ds(grp, "parent_idx", np.array(-1, dtype=np.int32), is_row=False)

    return dsets

def append_record(dsets, state, result, parent_id):
    """Append one row for all tracked fields."""
    edge_list = parse_edgelist(state).astype(np.uint32)

    ler     = float(result['logical_error_rates'][0]) if len(result['logical_error_rates']) else 0.0
    lstd    = float(result['stderrs'][0]) if len(result['stderrs']) else 0.0
    runtime = float(result['runtimes'][0]) if len(result['runtimes']) else 0.0

    _append_row(dsets["states"], edge_list)
    _append_row(dsets["logical_error_rates"],  np.array(ler, dtype=np.float64))
    _append_row(dsets["logical_error_rates_std"], np.array(lstd, dtype=np.float64))

    _append_row(dsets["n_classical"],   np.array(int(result['n_classical']), dtype=np.int32))
    _append_row(dsets["n_classical_T"], np.array(int(result['n_T_classical']), dtype=np.int32))
    _append_row(dsets["k_classical"],   np.array(int(result['k_classical']), dtype=np.int32))
    _append_row(dsets["k_classical_T"], np.array(int(result['k_T_classical']), dtype=np.int32))
    _append_row(dsets["rank_H"],        np.array(int(result['rank_H']), dtype=np.int32))
    _append_row(dsets["skipped"],       np.array(int(result['skipped']), dtype=np.int32))

    _append_row(dsets["distances_classical"],   np.array(float(result['d_classical']), dtype=np.float64))
    _append_row(dsets["distances_classical_T"], np.array(float(result['d_T_classical']), dtype=np.float64))
    _append_row(dsets["distances_quantum"],     np.array(float(result['d_quantum']), dtype=np.float64))
    _append_row(dsets["distances_Hx"],          np.array(float(result['d_Hx']), dtype=np.float64))
    _append_row(dsets["distances_Hz"],          np.array(float(result['d_Hz']), dtype=np.float64))
    _append_row(dsets["decoding_runtimes"],     np.array(runtime, dtype=np.float64))

    _append_row(dsets["parent_idx"], int(parent_id))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    parser.add_argument('-N', action="store", dest='N', default=None, type=int, required=False)
    parser.add_argument('-L', action="store", dest='L', default=None, type=int, required=False)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float, required=False)
    args = parser.parse_args()

    C = args.C
    N, L = exploration_params[C] if (args.N is None or args.L is None) else (args.N, args.L)
    p = noise_levels[C] if args.p is None else args.p
    print(f"{C = }, {N = }, {L = }, {p = }, {BEAM_WIDTH = }")

    osd_order = 2
    ms_scaling_factor = 0.625

    # Load Initial State
    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    start_time = time.time()
    
    with h5py.File(output_file, "a") as f:
        grp = f.require_group(codes[C])
        grp.attrs['MC_budget'] = MC_budget
        grp.attrs['p'] = p
        grp.attrs['N'] = N
        grp.attrs['L'] = L
        grp.attrs['BEAM_WIDTH'] = BEAM_WIDTH
        grp.attrs['run_label'] = "Beam Search"

        # --- Initial Evaluation ---
        cost_result = evaluate_performance_of_state(state=initial_state, p_vals=[p], MC_budget=MC_budget, run_label=run_label, canskip=False)
        
        init_dist = int(min(float(cost_result['d_classical']), float(cost_result['d_T_classical'])))
        distance_threshold = init_dist + 1
        logical_error_rate = float(cost_result['logical_error_rates'][0])
        min_cost = logical_error_rate

        # Initialize Datasets
        dsets = init_dsets(grp, initial_state, cost_result)
        append_record(dsets, initial_state, cost_result, parent_id=-1)

        _append_row(dsets["beam_survivors"], np.pad([0], (0, BEAM_WIDTH-1), constant_values=-1))
        
        # Save "Best State"
        best_state_edgelist = parse_edgelist(initial_state).astype(np.uint32)
        if "best_state" in grp: del grp["best_state"]
        grp.create_dataset("best_state", data=best_state_edgelist[np.newaxis, :], dtype=np.uint32)
        grp.attrs['min_cost'] = min_cost

        # --- BEAM INITIALIZATION ---
        current_beam = [{
            'state': initial_state,
            'cost': logical_error_rate,
            'result': cost_result,
            'row_idx': 0,
            'dist': init_dist
        }]

        print(f"Starting Beam Search with Width {BEAM_WIDTH}...")
        print(f"Initial Cost: {logical_error_rate:.6f}, Distance Threshold: {distance_threshold}")

        # --- MAIN LOOP (Depth L) ---
        for l in range(L):
            print("="*60)
            
            # 1. ADJUST BUDGET: Divide N by current beam size
            # n_per_parent = max(1, N // len(current_beam))
            n_per_parent = N - 1
            
            print(f"Iteration {l+1}/{L} | Beam Size: {len(current_beam)} | Neighbors per parent: {n_per_parent}")

            step_start_row = _current_stream_len(dsets)
            _flush_file(f)

            valid_candidates = []
            fallback_candidates = []
            scanned_total = 0

            # --- PROCESS CURRENT BEAM ---
            for parent_idx, parent_info in enumerate(current_beam):
                parent_state = parent_info['state']
                parent_row_idx = parent_info['row_idx']
                
                print(f"  >>> [Parent {parent_idx+1}/{len(current_beam)} ({parent_row_idx})] Starting scan...")
                
                valid_found_this_parent = 0

                # Scan neighbors for THIS parent
                for n in range(n_per_parent):
                    print(f"    - Scanning neighbor {n+1}/{n_per_parent} of parent {parent_idx+1} ({parent_row_idx}) in Iteration {l+1}...")
                    scanned_total += 1
                    
                    neighbor, _, _ = generate_neighbor_highlight(parent_state)
                    
                    cost_result = evaluate_performance_of_state(
                        state=neighbor, 
                        p_vals=[p], 
                        MC_budget=MC_budget, 
                        run_label=run_label, 
                        distance_threshold=distance_threshold
                    )

                    append_record(dsets, neighbor, cost_result, parent_id=parent_idx)
                    current_row_idx = _current_stream_len(dsets) - 1
                    
                    ler = float(cost_result['logical_error_rates'][0]) if len(cost_result['logical_error_rates']) else 1.0
                    dist_any = min(float(cost_result['d_classical']), float(cost_result['d_T_classical']))
                    skipped = cost_result['skipped']

                    # Global Best Check
                    if not skipped and ler < min_cost:
                        min_cost = ler
                        print(f"      [!] New Global Min: {min_cost:.6f}")
                        grp.attrs['min_cost'] = min_cost
                        if "best_state" in grp: del grp["best_state"]
                        best_state_edgelist = parse_edgelist(neighbor).astype(np.uint32)
                        grp.create_dataset("best_state", data=best_state_edgelist, shape=(1, best_state_edgelist.shape[0]), dtype=np.uint32)

                    # Collect Candidates
                    if not skipped:
                        valid_candidates.append({
                            'cost': ler, 'state': neighbor, 'result': cost_result,
                            'row_idx': current_row_idx, 'dist': dist_any
                        })
                        valid_found_this_parent += 1
                    else:
                        fallback_candidates.append({
                            'dist': dist_any, 'cost': ler, 'state': neighbor,
                            'result': cost_result, 'row_idx': current_row_idx
                        })

                    # Periodic Flush
                    if n % 5 == 0: _flush_file(f)

                    # 2. PER-PARENT EARLY STOP
                    if valid_found_this_parent >= EARLY_VALID_TARGET:
                        print(f"      [Parent {parent_idx+1}] Reached target ({EARLY_VALID_TARGET} valid). Stopping early.")
                        break
                
                print(f"  <<< [Parent {parent_idx+1}] Finished. Found {valid_found_this_parent} valid.")


            # --- SELECTION STEP ---
            next_beam = []

            candidate_pool = list(valid_candidates)
            if INCLUDE_PARENTS:
                candidate_pool.extend(current_beam)

            if candidate_pool:
                # Sort all candidates from ALL parents by cost
                candidate_pool.sort(key=lambda x: x['cost'])

                # Pick top K
                top_k = candidate_pool[:BEAM_WIDTH]
                next_beam = top_k
                
                costs_str = ", ".join([f"{c['cost']:.5f}" for c in top_k])
                sources = ["(Parent)" if c['row_idx'] < step_start_row else "(Child)" for c in top_k]
                print(f"  Selection: Keeping {len(top_k)} best. Costs: [{costs_str}]")
                print(f"  Sources: {sources}")
                print(f"  Distances: {[int(c['dist']) for c in top_k]}")

                distance_threshold = max(top_k, key=lambda x: x['dist'])['dist']

            else:
                # Fallback logic
                print("  No valid neighbors found. Attempting fallback (max distance)...")
                if fallback_candidates:
                    fallback_candidates.sort(key=lambda x: x['dist'], reverse=True)
                    best_fallback = fallback_candidates[0]
                    
                    # Force evaluate if it was skipped
                    if best_fallback['result'].get('skipped', False):
                        print("    Forcing evaluation of best fallback...")
                        full_result = evaluate_performance_of_state(
                            state=best_fallback['state'], p_vals=[p], MC_budget=MC_budget, 
                            run_label=run_label, distance_threshold=distance_threshold, canskip=False
                        )
                        append_record(dsets, best_fallback['state'], full_result, parent_id=best_fallback['parent_idx'])
                        best_fallback['result'] = full_result
                        best_fallback['cost'] = float(full_result['logical_error_rates'][0])
                        best_fallback['row_idx'] = _current_stream_len(dsets) - 1

                    next_beam = [best_fallback]
                    distance_threshold = int(best_fallback['dist'])
                    print(f"    Fallback selected. New Threshold: {distance_threshold}")
                else:
                    print("  CRITICAL: Search exhausted.")
                    break

            current_beam = next_beam
            
            # NEW: Save the row indices of the selected beam survivors
            survivor_indices = [c['row_idx'] for c in next_beam]
            # Pad with -1 if fewer survivors than BEAM_WIDTH (e.g. fallback scenarios)
            while len(survivor_indices) < BEAM_WIDTH:
                survivor_indices.append(-1)
            
            _append_row(dsets["beam_survivors"], np.array(survivor_indices, dtype=np.int64))
            _append_row(dsets["step_summaries"], np.array([step_start_row, scanned_total, len(valid_candidates), survivor_indices[0]]))

            print(f"  Survivors: {survivor_indices}")

            print(f"Updated distance threshold to {int(distance_threshold)}")
            runtime_so_far = time.time() - start_time
            print(f"runtime so far: {runtime_so_far // 3600} hours {(runtime_so_far % 3600) // 60} minutes {runtime_so_far % 60} seconds")

        grp.attrs['total_runtime'] = time.time() - start_time
        _flush_file(f)
        print(f"Best code found: min cost = {min_cost:.6f}")
        print(f"Total runtime: {grp.attrs['total_runtime'] // 3600} hours {(grp.attrs['total_runtime'] % 3600) // 60} minutes {grp.attrs['total_runtime'] % 60} seconds")
        print(f"Done. Saved to {output_file}")