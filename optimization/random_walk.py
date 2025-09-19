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

output_file = "optimization/results/random_walk_with_distance.hdf5"

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

    # bp_max_iter = 4
    # osd_order = 60
    osd_order = 2
    ms_scaling_factor = 0.625

    # ----------------- Random Walk ------------------------------------------------
    states, logical_error_rates, stds = [], [], []

    # Initialize the rw with the corresponding initial state. 
    initial_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    # input_file = "optimization/results/random_walk.hdf5"
    # with h5py.File(input_file, 'r') as f:
    #     best_state_edge_list = f[codes[C]]['best_state'][:][0]
    # best_state = from_edgelist(best_state_edge_list)
    # print(f"See neighbors of the best state found from previous random walk")

    # initial_state = best_state

    print(f"Exploring code family {codes[C]} with {N} neighbors and {L} iterations.")

    decoding_runtimes = []
    min_cost = np.inf
    min_state = None

    best_gain = 0.0
    original_cost = None

    distances_classical = []
    distances_quantum = []

    start_time = time.time()

    # Random Walk loop:
    for l in range(L):
        # explore a random walk from the current state
        if l > 0:
            state, old_edges, new_edges = generate_neighbor_highlight(state)
            # print(f"Old edges: {old_edges}, New edges: {new_edges}")
        else:
            state = initial_state

        cost_result = evaluate_performance_of_state(state=state, p_vals=[p], MC_budget=MC_budget)
        
        logical_error_rate = cost_result['logical_error_rates'][0]
        decoding_runtime = cost_result['runtimes'][0]
        std = cost_result['stderrs'][0]

        distances_classical.append(cost_result['d_classical'])
        distances_quantum.append(cost_result['d_quantum'])
        decoding_runtimes.append(decoding_runtime)

        if l == 0: # initial state
            original_cost = cost_result['logical_error_rates'][0]

        print(f"Iteration {l+1}/{L}: cost of current state = {logical_error_rate:.6f}, decoding time = {decoding_runtime:.6f} seconds")

        if logical_error_rate < min_cost:
            min_cost = logical_error_rate
            min_state = state

            print(f"Found minimum in iteration {l+1}/{L}: {min_cost:.6f} in chosen state")
        
        states.append(parse_edgelist(state))
        logical_error_rates.append(logical_error_rate)
        stds.append(std)

        # Neighbor exploration
        for n in range(N-1):
            print(f"Exploring neighbor {n+1}/{N-1} of iteration {l+1}/{L}...")
            neighbor, old_edges, new_edges = generate_neighbor_highlight(state)
            # print(f"Old edges: {old_edges}, New edges: {new_edges}")

            cost_result = evaluate_performance_of_state(state=neighbor, p_vals=[p], MC_budget=MC_budget)
            logical_error_rate = cost_result['logical_error_rates'][0]
            decoding_runtime = cost_result['runtimes'][0]
            std = cost_result['stderrs'][0]

            distances_classical.append(cost_result['d_classical'])
            distances_quantum.append(cost_result['d_quantum'])
            decoding_runtimes.append(decoding_runtime)

            if logical_error_rate < min_cost:
                min_cost = logical_error_rate
                min_state = neighbor

                print(f"Found minimum in neighbor {n+1}/{N-1} of iteration {l+1}/{L}: {min_cost:.6f}")

            states.append(parse_edgelist(neighbor))
            logical_error_rates.append(logical_error_rate)
            stds.append(std)
    
    print(f"Minimum cost found: {min_cost:.6f} in state {parse_edgelist(min_state)}")

    # Exploration finished: store results in hdf5 file
    states = np.row_stack(states, dtype=np.uint8)
    logical_error_rates = np.row_stack(logical_error_rates, dtype=np.float64)

    best_gain = original_cost / min_cost

    end_time = time.time()
    runtime = end_time - start_time

    avg_decoding_runtime = np.mean(decoding_runtimes)
    print(f"Average decoding runtime per state: {avg_decoding_runtime:.6f} seconds")

    print(f"Random exploration finished in {runtime:.2f} seconds.")

    with h5py.File(output_file, "a") as f:
        grp = f.require_group(codes[C])
        grp.attrs['MC_budget'] = MC_budget
        grp.attrs['p'] = p
        grp.attrs['total_runtime'] = runtime
        grp.attrs['avg_decoding_runtime'] = avg_decoding_runtime
        grp.attrs['original_cost'] = original_cost
        grp.attrs['min_cost'] = min_cost
        grp.attrs['best_gain'] = best_gain
        grp.attrs['N'] = N
        grp.attrs['L'] = L
        grp.attrs['osd_order'] = osd_order
        grp.attrs['ms_scaling_factor'] = ms_scaling_factor

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

        if "distances_classical" in grp:
            del grp["distances_classical"]
        grp.create_dataset("distances_classical", data=distances_classical)

        if "distances_quantum" in grp:
            del grp["distances_quantum"]
        grp.create_dataset("distances_quantum", data=distances_quantum)

        if "decoding_runtimes" in grp:
            del grp["decoding_runtimes"]
        grp.create_dataset("decoding_runtimes", data=decoding_runtimes)

    print(f"Results saved to {output_file} for code family {codes[C]}.")
