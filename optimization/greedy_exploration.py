import time
import numpy as np

import argparse
from tqdm import tqdm
import h5py
import sys
import os
import networkx as nx

from analyze_codes.decoder_performance_from_state import compute_decoding_performance_from_state

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_settings import generate_neighbor_highlight, load_tanner_graph, parse_edgelist
from experiments_settings import codes, path_to_initial_codes, textfiles
from experiments_settings import MC_budget, noise_levels

from state import State

# output_file = "optimization/results/greedy_exploration.hdf5"

if __name__ == '__main__':
    # Parse args: basically just a flag indicating the code family to explore. 
    # Optionally: args for the noise level to choose the cost function, 
    # the number of neighbors to explore, the length of the random walk. 
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    parser.add_argument('-N', action="store", dest='N', default=20, type=int, required=False)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float, required=False)
    parser.add_argument('-o', action="store", dest='output_file', default="optimization/results/greedy_exploration.hdf5", type=str, required=False)
    args = parser.parse_args()

    # Choose the code family
    C = args.C
    N = args.N  # number of iterations
    # Set the noise level
    p = noise_levels[C] if args.p is None else args.p
    output_file = args.output_file
    print(f"Exploring code family {codes[C]} with {N} iterations and {p} noise level with MC_budget = {MC_budget}.")

    bp_max_iter = 4
    # osd_order = 60
    osd_order = 2
    ms_scaling_factor = 0.625

    # ----------------- Greedy Exploration ------------------------------------------------
    states, logical_error_rates, stds = [], [], []

    # Initialize the rw with the corresponding initial state. 
    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    min_cost = np.inf
    min_state = None

    best_gain = 0.0
    original_cost = None

    start_time = time.time()

    cost_fn = lambda state: compute_decoding_performance_from_state(
        state=state, p_vals=[p], MC_budget=MC_budget, bp_max_iter=bp_max_iter, run_label=f"Greedy exploration {codes[C]}"
    )

    initial_cost_result = cost_fn(initial_state)

    current = State(initial_state, {
        'logical_error_rate': initial_cost_result['logical_error_rates'][0],
        'std': initial_cost_result['stds'][0],
        'runtime': initial_cost_result['runtimes'][0]
    })
    neighbor = State(nx.MultiGraph(), {})

    original_cost = current.cost_result['logical_error_rate']

    # greedy exploration loop:
    for i in range(N):
        print(f"Iteration {i+1}/{N}: cost of current state = {current.cost_result['logical_error_rate']:.6f}, decoding time = {current.cost_result['runtime']:.6f} seconds")

        if current.cost_result['logical_error_rate'] < min_cost:
            min_cost = current.cost_result['logical_error_rate']
            min_state = current.state

            print(f"Found minimum in iteration {i+1}/{N}: {min_cost:.6f} in chosen state")

        states.append(parse_edgelist(current.state))
        logical_error_rates.append(current.cost_result['logical_error_rate'])
        stds.append(current.cost_result['std'])

        if i < N - 1:  # if not the last iteration
            # choose a random neighbor from the current state
            neighbor_state, old_edges, new_edges = generate_neighbor_highlight(current.state)
            # print(f"Old edges: {old_edges}, New edges: {new_edges}")

            neighbor_cost_result = cost_fn(neighbor_state)
            neighbor.state = neighbor_state
            neighbor.cost_result = {
                'logical_error_rate': neighbor_cost_result['logical_error_rates'][0],
                'std': neighbor_cost_result['stds'][0],
                'runtime': neighbor_cost_result['runtimes'][0]
            }
            
            # If the neighbor is better than the current state, we choose it as the next state
            if neighbor.cost_result['logical_error_rate'] < current.cost_result['logical_error_rate']:
                current.state = neighbor.state
                current.cost_result = neighbor.cost_result
                print(f"Found better neighbor {i+1}/{N} so it would be the next state: {current.cost_result['logical_error_rate']:.6f}")
        
    print(f"Minimum cost found: {min_cost:.6f} in state {parse_edgelist(min_state)}")

    # Exploration finished: store results in hdf5 file
    states = np.row_stack(states, dtype=np.uint8)
    logical_error_rates = np.row_stack(logical_error_rates, dtype=np.float64)

    best_gain = original_cost / min_cost if min_cost > 0 else 0.0

    end_time = time.time()
    runtime = end_time - start_time

    print(f"Random exploration finished in {runtime:.2f} seconds.")

    with h5py.File(output_file, "a") as f:
        grp = f.require_group(codes[C])
        grp.attrs['MC_budget'] = MC_budget
        grp.attrs['p'] = p
        grp.attrs['runtime'] = runtime
        grp.attrs['original_cost'] = original_cost
        grp.attrs['min_cost'] = min_cost
        grp.attrs['best_gain'] = best_gain
        grp.attrs['num_iterations'] = N

        if "best_state" in grp:
            del grp["best_state"]
        grp.create_dataset("best_state", data=parse_edgelist(min_state), shape=(1, parse_edgelist(min_state).shape[0]))
        
        if "states" in grp:
            del grp["states"]
        grp.create_dataset("states", data=states)

        if "logical_error_rates" in grp:
            del grp["logical_error_rates"]
        grp.create_dataset("logical_error_rates", data=logical_error_rates)

        if "logical_error_rates_std" in grp:
            del grp["logical_error_rates_std"]
        grp.create_dataset("logical_error_rates_std", data=stds)

    print(f"Results saved to {output_file}")
    print(f"Exploration finished for code family {codes[C]}.")
