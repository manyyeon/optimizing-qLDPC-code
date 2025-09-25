import time
import numpy as np

import argparse
from tqdm import tqdm
import h5py
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.analyze_codes.decoder_performance_from_state import evaluate_performance_of_state
from optimization.experiments_settings import generate_neighbor_highlight, from_edgelist, load_tanner_graph, parse_edgelist
from optimization.experiments_settings import codes, path_to_initial_codes, textfiles
from optimization.experiments_settings import MC_budget, noise_levels
from optimization.compute_code_parameters import compute_code_parameters

# exploration_params = [(24, 120), (15, 70), (12, 40), (8, 30)]
exploration_params = [(24, 40), (15, 70), (12, 40), (12, 40)]

output_file = "optimization/results/best_neighbor_search_3.hdf5"

def _ensure_ds(grp, name, sample, is_row=True):
    """Create a resizable dataset if it doesn't exist; return the dataset."""
    if name in grp:
        return grp[name]
    arr = np.asarray(sample)
    if arr.ndim == 0:
        # scalar stream -> 1D [None]
        shape = (0,)
        maxshape = (None,)
        dtype = arr.dtype
    elif is_row:
        # row-stream -> 2D [None, D]
        shape = (0, arr.shape[0])
        maxshape = (None, arr.shape[0])
        dtype = arr.dtype
    else:
        # generic grow-on-first-axis
        shape = (0,) + arr.shape
        maxshape = (None,) + arr.shape
        dtype = arr.dtype
    return grp.create_dataset(name, shape=shape, maxshape=maxshape,
                              dtype=dtype, chunks=True)

def _append_row(ds, row):
    row = np.asarray(row)
    if ds.ndim == 1:
        # scalar stream
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = row
    else:
        # row stream (2D): last dim must match
        if ds.shape[1] != row.shape[0]:
            raise ValueError(f"Shape mismatch: ds has width {ds.shape[1]} but row has {row.shape[0]}")
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1, :] = row

def _flush_file(f):
    f.flush()
    # Best-effort fsync (may not be available on all platforms/drivers)
    try:
        fid = f.id.get_vfd_handle()  # h5py low-level handle
        if isinstance(fid, int) and fid >= 0:
            os.fsync(fid)
    except Exception:
        pass

def init_dsets(grp, initial_state, initial_result):
    """Create (or fetch) all datasets and return a dict name->dataset."""
    # infer dtypes (distances float64 for ∞; sizes/ranks int32; edges uint32)
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

    # distances & runtimes as float64 (∞ allowed)
    dsets["distances_classical"]   = _ensure_ds(grp, "distances_classical",   np.array(float(initial_result['d_classical']), dtype=np.float64), is_row=False)
    dsets["distances_classical_T"] = _ensure_ds(grp, "distances_classical_T", np.array(float(initial_result['d_T_classical']), dtype=np.float64), is_row=False)
    dsets["distances_quantum"]     = _ensure_ds(grp, "distances_quantum",     np.array(float(initial_result['d_quantum']), dtype=np.float64), is_row=False)
    dsets["distances_Hx"]          = _ensure_ds(grp, "distances_Hx",          np.array(float(initial_result['d_Hx']), dtype=np.float64), is_row=False)
    dsets["distances_Hz"]          = _ensure_ds(grp, "distances_Hz",          np.array(float(initial_result['d_Hz']), dtype=np.float64), is_row=False)
    dsets["decoding_runtimes"]     = _ensure_ds(grp, "decoding_runtimes",     np.array(float(initial_result['runtimes'][0]), dtype=np.float64), is_row=False)
    return dsets

def append_record(dsets, state, result):
    """Append one row for all tracked fields."""
    edge_list = parse_edgelist(state).astype(np.uint32)

    # Extract scalars safely
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

if __name__ == '__main__':
    # Parse args: basically just a flag indicating the code family to explore. 
    # Optionally: args for the noise level to choose the cost function, 
    # the number of neighbors to explore, the length of the random walk. 
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    parser.add_argument('-N', action="store", dest='N', default=None, type=int, required=False)
    parser.add_argument('-L', action="store", dest='L', default=None, type=int, required=False)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float, required=False)
    parser.add_argument('-d', action="store", dest='distance_threshold', default=8, type=int, required=False)
    args = parser.parse_args()

    # Choose the code family
    C = args.C
    # Set the number of neighbors and the length of the random walk
    N, L = exploration_params[C] if (args.N is None or args.L is None) else (args.N, args.L)
    # Set the noise level
    p = noise_levels[C] if args.p is None else args.p
    distance_threshold = args.distance_threshold
    print(f"{C = }, {N = }, {L = }, {p = }, {distance_threshold = }")



    if N < 2:
        raise ValueError("N (number of neighbors) must be at least 2 to select the best neighbor.")

    # bp_max_iter = 4
    # osd_order = 60
    osd_order = 2
    ms_scaling_factor = 0.625

    # ----------------- Random Walk ------------------------------------------------
    states, logical_error_rates, stds = [], [], []

    # Initialize the rw with the corresponding initial state.
    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    print(f"Exploring code family {codes[C]} with {N} neighbors and {L} iterations at p={p}... with distance threshold {distance_threshold}")

    decoding_runtimes = []
    min_cost = np.inf
    min_state = None

    distances_classical = []
    distances_quantum = []

    with h5py.File(output_file, "a") as f:
        grp = f.require_group(codes[C])
        # Save run-level attrs right away (and update later if needed)
        grp.attrs['MC_budget'] = MC_budget
        grp.attrs['p'] = p
        grp.attrs['N'] = N
        grp.attrs['L'] = L
        grp.attrs['osd_order'] = osd_order
        grp.attrs['ms_scaling_factor'] = ms_scaling_factor
        grp.attrs['total_runtime'] = 0.0
        grp.attrs['avg_decoding_runtime'] = 0.0

        ds_states = None                  # 2D: [num_samples, E] (edge list length)
        ds_lers = None                    # 2D: [num_samples, 1] or we can store 1D—below we use 1D
        ds_stds = None                    # 1D
        ds_dclass = None                  # 1D
        ds_dquant = None                  # 1D
        ds_runtimes = None                # 1D

        min_state_edgelist = None
        original_cost = None
        decoding_runtimes = []  # only for computing avg later
        start_time = time.time()

        cost_result = evaluate_performance_of_state(state=initial_state, p_vals=[p], MC_budget=MC_budget, distance_threshold=distance_threshold, canskip=False)
        logical_error_rate = float(cost_result['logical_error_rates'][0])
        initial_rank = cost_result['rank_H']
        print(f"Initial state logical error rate: {logical_error_rate:.6f}, rank_H: {initial_rank}")

        dsets = init_dsets(grp, initial_state, cost_result)
        append_record(dsets, initial_state, cost_result)

        grp.attrs['original_cost'] = logical_error_rate
        decoding_runtimes.append(float(cost_result['runtimes'][0]))

        min_cost = logical_error_rate
        best_state_edgelist = parse_edgelist(initial_state).astype(np.uint32)
        grp.attrs['min_cost'] = min_cost
        if "best_state" in grp:
            del grp["best_state"]
        grp.create_dataset("best_state", data=best_state_edgelist[np.newaxis, :], dtype=np.uint32)

        # Random Walk loop:
        for l in range(L):
            if l == 0:
                state = initial_state
                current_cost = logical_error_rate  # from the initial evaluation above
            else:
                state = next_state  # chosen from the previous iteration

            print(f"Iteration {l+1}/{L}: cost of current state = {current_cost:.6f}")

            # Ensure data hits disk every time (or do this every few appends for speed)
            _flush_file(f)

            # ---------- Neighbor exploration ----------
            best_neighbor = {'state': None, 'cost': np.inf, 'idx': -1, 'edge_list': None, 'result': None}
            for n in range(N-1):
                print("-"*50)    
                print(f"Exploring neighbor {n+1}/{N-1} of iteration {l+1}/{L}...")
                neighbor, old_edges, new_edges = generate_neighbor_highlight(state)
                cost_result = evaluate_performance_of_state(state=neighbor, p_vals=[p], MC_budget=MC_budget, distance_threshold=distance_threshold, initial_rank=initial_rank)

                append_record(dsets, neighbor, cost_result)

                logical_error_rate = cost_result['logical_error_rates'][0]
                decoding_runtime = cost_result['runtimes'][0]

                decoding_runtimes.append(decoding_runtime)

                if cost_result['skipped']:
                    grp.attrs['avg_decoding_runtime_so_far'] = float(np.mean(decoding_runtimes))
                    _flush_file(f)

                    print(f"Skipping neighbor {n+1}/{N-1} due to distance or rank criteria.")
                    continue

                # Track the best neighbor to continue the random walk
                if logical_error_rate < best_neighbor['cost']:
                    best_neighbor['cost'] = logical_error_rate
                    best_neighbor['state'] = neighbor
                    best_neighbor['idx'] = n
                    best_neighbor['edge_list'] = parse_edgelist(neighbor).astype(np.uint32)
                    best_neighbor['result'] = cost_result

                # Update minima + attrs immediately
                if logical_error_rate < min_cost:
                    min_cost = logical_error_rate
                    best_state = neighbor
                    print(f"Found minimum in neighbor {n+1}/{N-1} of iteration {l+1}/{L}: {min_cost:.6f}")
                    grp.attrs['min_cost'] = min_cost
                    if "best_state" in grp:
                        del grp["best_state"]
                    best_state_edgelist = parse_edgelist(best_state).astype(np.uint32)
                    grp.create_dataset("best_state", data=best_state_edgelist, shape=(1, best_state_edgelist.shape[0]), dtype=np.uint32)

                grp.attrs['avg_decoding_runtime_so_far'] = float(np.mean(decoding_runtimes))
                _flush_file(f)

                print(f"total runtime so far: {(time.time() - start_time)//3600}h {(time.time() - start_time)%3600//60}m {(time.time() - start_time)%60}s, avg decoding runtime so far: {grp.attrs['avg_decoding_runtime_so_far']:.4f}s")

            print(f"Best neighbor in iteration {l+1}/{L} has cost {best_neighbor['cost']:.6f}")
            # Prepare for next iteration
            if best_neighbor['state'] is None:
                print("No better neighbor found; starting next iteration with random state.")
                next_state, _, _ = generate_neighbor_highlight(state)
                cost_result_next_initial = evaluate_performance_of_state(state=next_state, p_vals=[p], MC_budget=MC_budget, distance_threshold=distance_threshold, initial_rank=initial_rank, canskip=False)

                # record the next state as the initial state of the next iteration
                append_record(dsets, next_state, cost_result_next_initial)

            else:
                next_state = best_neighbor['state']
                current_cost = best_neighbor['cost']

                # record the best neighbor as the initial state of the next iteration
                append_record(dsets, next_state, cost_result)

            if l == L-1 and n == N-2:
                print("Last iteration and last neighbor reached; not appending best neighbor again.")
                break
            
            _flush_file(f)


        # ---------- After loops: finish summary attrs ----------
        total_runtime = time.time() - start_time
        grp.attrs['total_runtime'] = total_runtime
        grp.attrs['avg_decoding_runtime'] = float(np.mean(decoding_runtimes))

        _flush_file(f)

    print(f"Streaming results saved to {output_file} for code family {codes[C]}.")

