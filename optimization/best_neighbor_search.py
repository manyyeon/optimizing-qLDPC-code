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
from optimization.experiments_settings import tanner_graph_to_parity_check_matrix
from optimization.experiments_settings import codes, path_to_initial_codes, textfiles
from optimization.experiments_settings import MC_budget, noise_levels
from optimization.compute_code_parameters import compute_code_parameters

# exploration_params = [(24, 120), (15, 70), (12, 40), (8, 30)]
exploration_params = [(50, 10), (50, 10), (50, 10), (50, 10)] # (N, L)

output_file = os.environ.get('OUTPUT_HDF5', "optimization/results/best_neighbor_search_5.hdf5")

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
    sig_cnt = int(len(signatures))

    check_patterns = [tuple(np.flatnonzero(H[i, :]).tolist()) for i in range(H.shape[0])]
    var_patterns = [tuple(np.flatnonzero(H[:, j]).tolist()) for j in range(H.shape[1])]
    check_cyclic_classes = len({_canonical_shift(sig, H.shape[1]) for sig in check_patterns})
    var_cyclic_classes = len({_canonical_shift(sig, H.shape[0]) for sig in var_patterns})

    return {
        "row_weights": row_w,
        "col_weights": col_w,
        "distinct_col_signatures": sig_cnt,
        "distinct_row_signatures": len({_canonical_shift(sig, H.shape[0]) for sig in check_patterns}),
        "check_cyclic_classes": check_cyclic_classes,
        "var_cyclic_classes": var_cyclic_classes,
        "row_weight_var": row_var,
        "col_weight_var": col_var,
        "row_degree_entropy": row_entropy,
        "col_degree_entropy": col_entropy,
    }


def compute_regularity_metrics_from_state(state) -> dict:
    H = tanner_graph_to_parity_check_matrix(state)
    return compute_regularity_metrics(H)


def _format_reg_metrics(metrics: dict) -> str:
    return (
        f"row_weights={metrics['row_weights']}, col_weights={metrics['col_weights']}, "
        f"sig_cnt={metrics['distinct_col_signatures']}, "
        f"row_sig_cnt={metrics['distinct_row_signatures']}, "
        f"check_classes={metrics['check_cyclic_classes']}, var_classes={metrics['var_cyclic_classes']}, "
        f"row_var={metrics['row_weight_var']:.3f}, col_var={metrics['col_weight_var']:.3f}, "
        f"row_ent={metrics['row_degree_entropy']:.3f}, col_ent={metrics['col_degree_entropy']:.3f}"
    )

def init_dsets(grp, initial_state, initial_result, initial_metrics):
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
    dsets["distances_Hx"]          = _ensure_ds(grp, "distances_Hx",          np.array(float(initial_result.get('d_Hx', 0.0)), dtype=np.float64), is_row=False)
    dsets["distances_Hz"]          = _ensure_ds(grp, "distances_Hz",          np.array(float(initial_result.get('d_Hz', 0.0)), dtype=np.float64), is_row=False)
    dsets["decoding_runtimes"]     = _ensure_ds(grp, "decoding_runtimes",     np.array(float(initial_result['runtimes'][0]), dtype=np.float64), is_row=False)
    
    dsets["wall_times"]            = _ensure_ds(grp, "wall_times",            np.array(0.0, dtype=np.float64), is_row=False)
    dsets["ler_wall_times"]        = _ensure_ds(grp, "ler_wall_times",        np.array(0.0, dtype=np.float64), is_row=False)

    dsets["row_weight_var"]        = _ensure_ds(grp, "row_weight_var",        np.array(float(initial_metrics['row_weight_var']), dtype=np.float64), is_row=False)
    dsets["col_weight_var"]        = _ensure_ds(grp, "col_weight_var",        np.array(float(initial_metrics['col_weight_var']), dtype=np.float64), is_row=False)
    dsets["distinct_col_signatures"] = _ensure_ds(grp, "distinct_col_signatures", np.array(int(initial_metrics['distinct_col_signatures']), dtype=np.int32), is_row=False)
    dsets["row_degree_entropy"]    = _ensure_ds(grp, "row_degree_entropy",    np.array(float(initial_metrics['row_degree_entropy']), dtype=np.float64), is_row=False)
    dsets["col_degree_entropy"]    = _ensure_ds(grp, "col_degree_entropy",    np.array(float(initial_metrics['col_degree_entropy']), dtype=np.float64), is_row=False)
    dsets["check_cyclic_classes"]  = _ensure_ds(grp, "check_cyclic_classes",  np.array(int(initial_metrics['check_cyclic_classes']), dtype=np.int32), is_row=False)
    dsets["var_cyclic_classes"]    = _ensure_ds(grp, "var_cyclic_classes",    np.array(int(initial_metrics['var_cyclic_classes']), dtype=np.int32), is_row=False)
    return dsets

def append_record(dsets, state, result, metrics, wall_time=np.nan):
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
    _append_row(dsets["distances_Hx"],          np.array(float(result.get('d_Hx', 0.0)), dtype=np.float64))
    _append_row(dsets["distances_Hz"],          np.array(float(result.get('d_Hz', 0.0)), dtype=np.float64))
    _append_row(dsets["decoding_runtimes"],     np.array(runtime, dtype=np.float64))
    _append_row(dsets["wall_times"],            np.array(wall_time, dtype=np.float64))
    _append_row(dsets["ler_wall_times"],        np.array(wall_time if ler > 0 else np.nan, dtype=np.float64))

    _append_row(dsets["row_weight_var"],        np.array(float(metrics['row_weight_var']), dtype=np.float64))
    _append_row(dsets["col_weight_var"],        np.array(float(metrics['col_weight_var']), dtype=np.float64))
    _append_row(dsets["distinct_col_signatures"], np.array(int(metrics['distinct_col_signatures']), dtype=np.int32))
    _append_row(dsets["row_degree_entropy"],    np.array(float(metrics['row_degree_entropy']), dtype=np.float64))
    _append_row(dsets["col_degree_entropy"],    np.array(float(metrics['col_degree_entropy']), dtype=np.float64))
    _append_row(dsets["check_cyclic_classes"],  np.array(int(metrics['check_cyclic_classes']), dtype=np.int32))
    _append_row(dsets["var_cyclic_classes"],    np.array(int(metrics['var_cyclic_classes']), dtype=np.int32))

if __name__ == '__main__':
    # Parse args: basically just a flag indicating the code family to explore. 
    # Optionally: args for the noise level to choose the cost function, 
    # the number of neighbors to explore, the length of the random walk. 
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    parser.add_argument('-N', action="store", dest='N', default=None, type=int, required=False)
    parser.add_argument('-L', action="store", dest='L', default=None, type=int, required=False)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float, required=False)
    parser.add_argument('-d', action="store", dest='distance_threshold', default=6, type=int, required=False)
    parser.add_argument('--target-distance', action="store", dest='target_distance', default=None, type=int, required=False)
    parser.add_argument('--mc-budget', type=int, default=10000, help='Monte Carlo budget')
    parser.add_argument('--stop-at-target', action='store_true',
                        help='Stop the run immediately after reaching target distance.')
    args = parser.parse_args()

    # Choose the code family
    C = args.C
    # Set the number of neighbors and the length of the random walk
    N, L = exploration_params[C] if (args.N is None or args.L is None) else (args.N, args.L)
    # Set the noise level
    p = noise_levels[C] if args.p is None else args.p
    distance_threshold = args.distance_threshold
    target_distance = args.target_distance
    stop_at_target = args.stop_at_target
    print(f"{C = }, {N = }, {L = }, {p = }, {distance_threshold = }")

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
        grp.attrs['distance_threshold'] = distance_threshold
        grp.attrs['target_distance'] = -1 if target_distance is None else target_distance
        grp.attrs['stop_at_target'] = bool(stop_at_target)

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

        cost_result = evaluate_performance_of_state(state=initial_state, p_vals=[p], MC_budget=args.mc_budget, distance_threshold=distance_threshold, canskip=False)
        logical_error_rate = float(cost_result['logical_error_rates'][0])
        initial_rank = cost_result['rank_H']
        print(f"Initial state logical error rate: {logical_error_rate:.6f}, rank_H: {initial_rank}")

        initial_metrics = compute_regularity_metrics_from_state(initial_state)
        min_dist_init = min(cost_result['d_classical'], cost_result['d_T_classical'])
        print(f"Initial distance: {min_dist_init}. Regularity: {_format_reg_metrics(initial_metrics)}")

        dsets = init_dsets(grp, initial_state, cost_result, initial_metrics)
        append_record(dsets, initial_state, cost_result, initial_metrics, time.time() - start_time)

        grp.attrs['original_cost'] = logical_error_rate
        decoding_runtimes.append(float(cost_result['runtimes'][0]))

        min_cost = logical_error_rate
        best_state_edgelist = parse_edgelist(initial_state).astype(np.uint32)
        grp.attrs['min_cost'] = min_cost
        if "best_state" in grp:
            del grp["best_state"]
        grp.create_dataset("best_state", data=best_state_edgelist[np.newaxis, :], dtype=np.uint32)

        target_reached = False
        best_dist = min_dist_init
        if target_distance is not None and min_dist_init >= target_distance:
            grp.attrs['time_to_target'] = time.time() - start_time
            grp.attrs['target_reached'] = True
            target_reached = True
            if stop_at_target:
                print(f"Target distance {target_distance} reached at initial state. Stopping early.")
                grp.attrs['total_runtime'] = float(time.time() - start_time)
                _flush_file(f)
                sys.exit()

        # Random Walk loop:
        for l in range(L):
            if l == 0:
                state = initial_state
                current_cost = logical_error_rate  # from the initial evaluation above
            else:
                state = next_state  # chosen from the previous iteration

            print("-"*50)
            print(f"Iteration {l+1}/{L}: cost of current state = {current_cost:.6f}")

            # Ensure data hits disk every time (or do this every few appends for speed)
            _flush_file(f)

            # ---------- Neighbor exploration ----------
            best_neighbor = {'state': None, 'cost': np.inf, 'idx': -1, 'edge_list': None, 'result': None, 'metrics': None}
            for n in range(N-1):
                print("-"*50)    
                print(f"Exploring neighbor {n+1}/{N-1} of iteration {l+1}/{L}...")
                neighbor, old_edges, new_edges = generate_neighbor_highlight(state)
                cost_result = evaluate_performance_of_state(state=neighbor, p_vals=[p], MC_budget=args.mc_budget, distance_threshold=distance_threshold)
                metrics = compute_regularity_metrics_from_state(neighbor)

                append_record(dsets, neighbor, cost_result, metrics, time.time() - start_time)

                logical_error_rate = cost_result['logical_error_rates'][0]
                decoding_runtime = cost_result['runtimes'][0]

                decoding_runtimes.append(decoding_runtime)

                min_dist = min(cost_result['d_classical'], cost_result['d_T_classical'])
                if min_dist > best_dist:
                    best_dist = min_dist
                    print(f"  New max dist {best_dist}. Regularity: {_format_reg_metrics(metrics)}")

                if (not target_reached) and target_distance is not None:
                    if min_dist >= target_distance:
                        grp.attrs['time_to_target'] = time.time() - start_time
                        grp.attrs['target_reached'] = True
                        target_reached = True
                        if stop_at_target:
                            print(f"Target distance {target_distance} reached. Stopping early.")
                            grp.attrs['total_runtime'] = float(time.time() - start_time)
                            _flush_file(f)
                            sys.exit()

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
                    best_neighbor['metrics'] = metrics

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
            
            if l == L-1:
                print("Last iteration and last neighbor reached; not appending best neighbor again.")
                break

            # Prepare for next iteration
            if best_neighbor['state'] is None:
                print("No better neighbor found; starting next iteration with random state.")
                next_state, _, _ = generate_neighbor_highlight(state)
                cost_result_next_initial = evaluate_performance_of_state(state=next_state, p_vals=[p], MC_budget=args.mc_budget, distance_threshold=distance_threshold, initial_rank=initial_rank, canskip=False)
                metrics_next_initial = compute_regularity_metrics_from_state(next_state)

                # record the next state as the initial state of the next iteration
                append_record(dsets, next_state, cost_result_next_initial, metrics_next_initial, time.time() - start_time)

            else:
                next_state = best_neighbor['state']
                current_cost = best_neighbor['cost']

                # record the best neighbor as the initial state of the next iteration
                append_record(dsets, next_state, best_neighbor['result'], best_neighbor['metrics'], time.time() - start_time)
            
            _flush_file(f)


        # ---------- After loops: finish summary attrs ----------
        total_runtime = time.time() - start_time
        print(f"Total runtime: {total_runtime//3600}h {(total_runtime%3600)//60}m {(total_runtime%60):.2f}s ({total_runtime:.2f}s)")
        grp.attrs['total_runtime'] = total_runtime
        grp.attrs['avg_decoding_runtime'] = float(np.mean(decoding_runtimes))

        _flush_file(f)

    print(f"Streaming results saved to {output_file} for code family {codes[C]}.")

