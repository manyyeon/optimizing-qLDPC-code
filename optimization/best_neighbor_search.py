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

output_file = "optimization/results/best_neighbor_search.hdf5"

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

if __name__ == '__main__':
    # Parse args: basically just a flag indicating the code family to explore. 
    # Optionally: args for the noise level to choose the cost function, 
    # the number of neighbors to explore, the length of the random walk. 
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    parser.add_argument('-N', action="store", dest='N', default=None, type=int, required=False)
    parser.add_argument('-L', action="store", dest='L', default=None, type=int, required=False)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float, required=False)
    args = parser.parse_args()

    # Choose the code family
    C = args.C
    # Set the number of neighbors and the length of the random walk
    N, L = exploration_params[C] if (args.N is None or args.L is None) else (args.N, args.L)
    # Set the noise level
    p = noise_levels[C] if args.p is None else args.p
    print(f"{C = }, {N = }, {L = }, {p = }")

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

    print(f"Exploring code family {codes[C]} with {N} neighbors and {L} iterations.")

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

        cost_result = evaluate_performance_of_state(state=initial_state, p_vals=[p], MC_budget=MC_budget, canskip=False)
        logical_error_rate = float(cost_result['logical_error_rates'][0])
        grp.attrs['original_cost'] = logical_error_rate
        decoding_runtimes.append(float(cost_result['runtimes'][0]))

        min_cost = logical_error_rate
        best_state_edgelist = parse_edgelist(initial_state).astype(np.uint32)
        grp.attrs['min_cost'] = min_cost
        if "best_state" in grp:
            del grp["best_state"]
        grp.create_dataset("best_state", data=best_state_edgelist[np.newaxis, :], dtype=np.uint32)

        if ds_states is None:
            ds_states   = _ensure_ds(grp, "states",               parse_edgelist(initial_state).astype(np.uint32), is_row=True)
            ds_lers     = _ensure_ds(grp, "logical_error_rates",  np.array(logical_error_rate, dtype=np.float64),   is_row=False)  # 1D
            ds_stds     = _ensure_ds(grp, "logical_error_rates_std", np.array(cost_result['stderrs'][0], dtype=np.float64), is_row=False)
            ds_dclass   = _ensure_ds(grp, "distances_classical",  np.array(cost_result['d_classical'], dtype=np.int32),    is_row=False)
            ds_dquant   = _ensure_ds(grp, "distances_quantum",    np.array(cost_result['d_quantum'], dtype=np.int32),    is_row=False)
            ds_runtimes = _ensure_ds(grp, "decoding_runtimes",    np.array(cost_result['runtimes'][0], dtype=np.float64),    is_row=False)
            ds_dhx      = _ensure_ds(grp, "distances_Hx",         np.array(cost_result['d_Hx'], dtype=np.int32),       is_row=False)
            ds_dhz      = _ensure_ds(grp, "distances_Hz",         np.array(cost_result['d_Hz'], dtype=np.int32),       is_row=False)
            ds_skipped  = _ensure_ds(grp, "skipped", np.array(int(cost_result['skipped'])), is_row=False)

        _append_row(ds_states, parse_edgelist(initial_state).astype(np.uint32))
        _append_row(ds_lers, np.array(logical_error_rate, dtype=np.float64))
        _append_row(ds_stds, np.array(cost_result['stderrs'][0], dtype=np.float64))
        _append_row(ds_dclass, np.array(cost_result['d_classical'], dtype=np.int32))
        _append_row(ds_dquant, np.array(cost_result['d_quantum'], dtype=np.int32))
        _append_row(ds_dhx, np.array(cost_result['d_Hx'], dtype=np.int32))
        _append_row(ds_dhz, np.array(cost_result['d_Hz'], dtype=np.int32))
        _append_row(ds_runtimes, np.array(cost_result['runtimes'][0], dtype=np.float64))
        _append_row(ds_skipped, np.array(int(cost_result['skipped'])))
        
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
            cost_results = []
            best_neighbor = {'state': None, 'cost': np.inf, 'idx': -1, 'edge_list': None}
            for n in range(N-1):    
                print(f"Exploring neighbor {n+1}/{N-1} of iteration {l+1}/{L}...")
                neighbor, old_edges, new_edges = generate_neighbor_highlight(state)
                cost_result = evaluate_performance_of_state(state=neighbor, p_vals=[p], MC_budget=MC_budget)

                logical_error_rate = cost_result['logical_error_rates'][0]
                decoding_runtime = cost_result['runtimes'][0]

                # Append results
                _append_row(ds_states, parse_edgelist(neighbor).astype(np.uint32))
                _append_row(ds_lers, np.array(logical_error_rate, dtype=np.float64))
                _append_row(ds_stds, np.array(cost_result['stderrs'][0], dtype=np.float64))
                _append_row(ds_dclass, np.array(cost_result['d_classical'], dtype=np.int32))
                _append_row(ds_dquant, np.array(cost_result['d_quantum'], dtype=np.int32))
                _append_row(ds_dhx, np.array(cost_result['d_Hx'], dtype=np.int32))
                _append_row(ds_dhz, np.array(cost_result['d_Hz'], dtype=np.int32))
                _append_row(ds_runtimes, np.array(decoding_runtime, dtype=np.float64))
                _append_row(ds_skipped, np.array(int(cost_result['skipped'])))

                decoding_runtimes.append(decoding_runtime)

                if cost_result['skipped']:
                    grp.attrs['avg_decoding_runtime_so_far'] = float(np.mean(decoding_runtimes))
                    _flush_file(f)

                    print(f"Skipping neighbor {n+1}/{N-1} due to insufficient distance.")
                    continue

                # Track the best neighbor to continue the random walk
                if logical_error_rate < best_neighbor['cost']:
                    best_neighbor['cost'] = logical_error_rate
                    best_neighbor['state'] = neighbor
                    best_neighbor['idx'] = n
                    best_neighbor['edge_list'] = parse_edgelist(neighbor).astype(np.uint32)

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

            print(f"Best neighbor in iteration {l+1}/{L} has cost {best_neighbor['cost']:.6f}")
            # Prepare for next iteration
            next_state   = best_neighbor['state']
            current_cost = best_neighbor['cost']

            if l == L-1 and n == N-2:
                print("Last iteration and last neighbor reached; not appending best neighbor again.")
                break

            # Append best neighbor again to show as the initial state of next iteration
            _append_row(ds_states, best_neighbor['edge_list'])
            _append_row(ds_lers, np.array(best_neighbor['cost'], dtype=np.float64))
            _append_row(ds_stds, np.array(cost_results[best_neighbor['idx']]['stderrs'], dtype=np.float64))
            _append_row(ds_dclass, np.array(cost_results[best_neighbor['idx']]['d_classical'], dtype=np.int32))
            _append_row(ds_dquant, np.array(cost_results[best_neighbor['idx']]['d_quantum'], dtype=np.int32))
            _append_row(ds_dhx, np.array(cost_results[best_neighbor['idx']]['d_Hx'], dtype=np.int32))
            _append_row(ds_dhz, np.array(cost_results[best_neighbor['idx']]['d_Hz'], dtype=np.int32))
            _append_row(ds_runtimes, np.array(cost_results[best_neighbor['idx']]['runtimes'], dtype=np.float64))
            _append_row(ds_skipped, np.array(int(cost_results[best_neighbor['idx']]['skipped'])))

            _flush_file(f)


        # ---------- After loops: finish summary attrs ----------
        total_runtime = time.time() - start_time
        grp.attrs['total_runtime'] = total_runtime
        grp.attrs['avg_decoding_runtime'] = float(np.mean(decoding_runtimes))

        _flush_file(f)

    print(f"Streaming results saved to {output_file} for code family {codes[C]}.")

