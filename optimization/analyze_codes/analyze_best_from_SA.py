import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimization.experiments_settings import codes, from_edgelist, load_tanner_graph, parse_edgelist
from optimization.experiments_settings import MC_budget, p_vals, path_to_initial_codes, textfiles
from optimization.analyze_codes.decoder_performance_from_state import evaluate_performance_of_state
import h5py
import argparse
import numpy as np

grpname = codes
p_vals = np.logspace(-2, -1, 20)
print(f"p_vals: {p_vals}")
MC_budget = int(1e6)

# names = ["PEG_codes", "SA_codes", "PS_codes", "PE_codes"]
names = ["PEG_codes"]
input_file = "optimization/results/simulated_annealing_beta7.hdf5"
output_file = "optimization/results/analysis_best_from_simulated_annealing_beta7.hdf5"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    args = parser.parse_args()

    # Choose the code family
    C = args.C

    fn_data = {}

    with h5py.File(input_file, 'r') as f:
        best_state_edge_list = f[grpname[C]]['best_state'][:][0]
        index_of_min = np.argmin(f[grpname[C]]['logical_error_rates'])
        print(f"Minimum logical error rate found in state {index_of_min} with value {f[grpname[C]]['logical_error_rates'][index_of_min]}")
        best_state = from_edgelist(best_state_edge_list)

    cost_result = evaluate_performance_of_state(best_state, p_vals, MC_budget, 
                                                            bp_max_iter=None, run_label="Best state from Greedy Exploration")

    logical_error_rates = np.row_stack(cost_result['logical_error_rates'], dtype=np.float64)

    with h5py.File(output_file, "a") as f:
        grp = f.require_group(grpname[C])
        grp.attrs['MC_budget'] = MC_budget
        
        if 'physical_error_rates' in grp:
            del grp['physical_error_rates']
        grp.create_dataset("physical_error_rates", data=p_vals)

        if 'logical_error_rates' in grp:
            del grp['logical_error_rates']
        grp.create_dataset("logical_error_rates", data=logical_error_rates)

        if 'logical_error_rates_stderr' in grp:
            del grp['logical_error_rates_stderr']
        grp.create_dataset("logical_error_rates_stderr", data=cost_result['stderrs'])

        if 'runtimes' in grp:
            del grp['runtimes']
        grp.create_dataset("runtimes", data=cost_result['runtimes'])
    
    print(f"Analysis of best state from greedy exploration for code family {codes[C]} completed and saved to {output_file}.")

