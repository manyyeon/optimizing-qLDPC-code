import os
import sys
import time
import numpy as np
import numpy.random as npr

import networkx as nx

from typing import Callable
import argparse
from tqdm import tqdm
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_settings import generate_neighbor_highlight, load_tanner_graph, parse_edgelist, generate_neighbor
from experiments_settings import codes, path_to_initial_codes, textfiles
from experiments_settings import MC_budget, noise_levels


from optimization.analyze_codes.decoder_performance_from_state import compute_decoding_performance_from_state

from state import State

output_file_path = "optimization/results/simulated_annealing_beta7.hdf5"

def arctan_diff_schedule(t: int, coef: float=10.) -> float:
    return 1./(1 + coef*t**2)

def simulated_annealing(cost_fn: Callable, random_neighbor: Callable, schedule_function: Callable,
                        initial_state: nx.MultiGraph, max_iterations: int, epsilon: float=0.0) -> dict:
    """
    Executes the Simulated Annealing (SA) algorithm to minimize (optimize) a cost function 
    over the state space of Tanner graphs with fixed vertex sets and number of (multi)edges. 

    :param cost_function: function to be minimized.
    :param random_neighbor: function which returns a random neighbor of a given point.
    :param schedule_function: function which computes the temperature schedule.
    :param initial_state: initial state.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :param max_iterations: maximum number of iterations.
    :return history: history of points visited by the algorithm.
    :return cost_history: cost function values along the history.
    """

    initial_cost_result = cost_fn(initial_state)

    current = State(initial_state, {
        'logical_error_rate': initial_cost_result['logical_error_rates'][0],
        'stderr': initial_cost_result['stderrs'][0],
        'runtime': initial_cost_result['runtimes'][0]
    })
    neighbor = State(nx.MultiGraph(), {})
    min_state, min_cost, best_std = current.state, current.cost_result['logical_error_rate'], current.cost_result['stderr']

    states, logical_error_rates, stderrs, runtimes = [current.state], [current.cost_result['logical_error_rate']], [current.cost_result['stderr']], [current.cost_result['runtime']]
    original_cost = current.cost_result['logical_error_rate']
    for num_iterations in range(max_iterations):
        print(f"Iteration {num_iterations+1}/{max_iterations}: Current cost = {current.cost_result['logical_error_rate']}, Best cost = {min_cost}, Temperature = {schedule_function(num_iterations/max_iterations)}")
        if current.cost_result['logical_error_rate'] < epsilon:
            break
        
        temperature = schedule_function(num_iterations/max_iterations)

        neighbor_state, old_edges, new_edges = random_neighbor(current.state)

        neighbor_cost_result = cost_fn(neighbor_state)
        neighbor.state = neighbor_state
        neighbor.cost_result = {
            'logical_error_rate': neighbor_cost_result['logical_error_rates'][0],
            'stderr': neighbor_cost_result['stderrs'][0],
            'runtime': neighbor_cost_result['runtimes'][0]
        }


        delta_log_cost = np.log(neighbor.cost_result['logical_error_rate']) - np.log(current.cost_result['logical_error_rate'])

        if delta_log_cost < 0:
            # If neighbor's value is better, accept it
            current.state = neighbor.state
            current.cost_result = neighbor.cost_result

            min_state, min_cost, best_std = min([(current.state, current.cost_result['logical_error_rate'], current.cost_result['stderr']), (min_state, min_cost, best_std)], key=lambda p: p[1])
        else:
            # If neighbor's value is worse, accept it with a probability
            if npr.rand() < np.exp(-delta_log_cost/temperature):
                current.state = neighbor.state
                current.cost_result = neighbor.cost_result

                min_state, min_cost, best_std = min([(current.state, current.cost_result['logical_error_rate'], current.cost_result['stderr']), (min_state, min_cost, best_std)], key=lambda p: p[1])

        states.append(current.state)
        logical_error_rates.append(current.cost_result['logical_error_rate'])
        stderrs.append(current.cost_result['stderr'])
        runtimes.append(current.cost_result['runtime'])

    return {
        'states': states,
        'logical_error_rates': logical_error_rates,
        'stderrs': stderrs,
        'best_state': min_state,
        'min_cost': min_cost,
        'best_std': best_std,
        'original_cost': original_cost,
        'runtimes': runtimes,
    }


# sim_ann_params = {'max_iter': [2400, 900, 450, 180], 
#                   'beta': 4, 
#                   'betadl': [1, 4, 7, 10]}

sim_ann_params = {'max_iter': [2000, 900, 450, 180], 
                  'beta': 7, 
                  'betadl': [1, 4, 7, 10]}

if __name__ == '__main__':
    # Parse args: -C (Code family to optimize), -L (Length of the optimization i.e. max_iterations), 
    # -p (noise level for the cost function) 
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    parser.add_argument('-L', action="store", dest='max_iter', default=None, type=int)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float)
    args = parser.parse_args()

    # Choose the code family
    C = args.C
    p = noise_levels[C] if args.p is None else args.p

    # The code family already defines some preferred values for max_iterations, initial_state
    max_iter = sim_ann_params['max_iter'][C] if args.max_iter is None else args.max_iter

    print(f"Optimizing code family {codes[C]} with {max_iter} iterations and noise level {p:.4f}")
    
    start_time = time.time()

    initial_state = load_tanner_graph(path_to_initial_codes+textfiles[C])

    # Define cost and scheduling functions
    cost_fn = lambda s: compute_decoding_performance_from_state(s, [p], MC_budget, run_label=f"Simulated Annealing")
    sched_fn = lambda t: arctan_diff_schedule(t, coef=sim_ann_params['beta'])

    # Run Simulated Annealing
    sim_ann_res = simulated_annealing(cost_fn=cost_fn, random_neighbor=generate_neighbor_highlight, 
                                      schedule_function=sched_fn, initial_state=initial_state, max_iterations=max_iter)

    end_time = time.time()
    runtime = end_time - start_time
    # Store results in HDF5 file
    with h5py.File(output_file_path, "a") as f:
        grp = f.require_group(codes[C])
        grp.attrs['MC_budget'] = MC_budget
        grp.attrs['p'] = p
        grp.attrs['runtime'] = runtime
        grp.attrs['original_cost'] = sim_ann_res['original_cost']
        grp.attrs['min_cost'] = sim_ann_res['min_cost']
        grp.attrs['best_std'] = sim_ann_res['best_std']
        grp.attrs['best_gain'] = sim_ann_res['min_cost'] / sim_ann_res['original_cost']
        grp.attrs['num_iterations'] = max_iter

        if "best_state" in grp:
            del grp["best_state"]
        grp.create_dataset("best_state", data=parse_edgelist(sim_ann_res['best_state']), shape=(1, parse_edgelist(sim_ann_res['best_state']).shape[0]))

        if "states" in grp:
            del grp["states"]
        grp.create_dataset("states", data=sim_ann_res['states'])

        if "logical_error_rates" in grp:
            del grp["logical_error_rates"]
        grp.create_dataset("logical_error_rates", data=sim_ann_res['logical_error_rates'])

        if "logical_error_rates_stderr" in grp:
            del grp["logical_error_rates_stderr"]
        grp.create_dataset("logical_error_rates_stderr", data=sim_ann_res['stderrs'])

        if "runtimes" in grp:
            del grp["runtimes"]
        grp.create_dataset("runtimes", data=sim_ann_res['runtimes'])

    print(f"Results saved to {output_file_path}")
    print(f"Exploration finished for code family {codes[C]}.")