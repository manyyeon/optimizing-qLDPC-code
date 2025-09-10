import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimization.experiments_settings import codes, load_tanner_graph, parse_edgelist
from optimization.experiments_settings import MC_budget, p_vals, path_to_initial_codes, textfiles
from optimization.analyze_codes.decoder_performance_from_state import compute_decoding_performance_from_state
import h5py
import argparse
import numpy as np

grpname = codes
p_vals = np.logspace(-2, -1, 20)
MC_budget = int(1e5)

# names = ["PEG_codes", "SA_codes", "PS_codes", "PE_codes"]
names = ["PEG_codes"]
output_file = "optimization/results/analysis_original_state_2.hdf5"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    args = parser.parse_args()

    # Choose the code family
    C = args.C

    fn_data = {}

    original_state = load_tanner_graph(path_to_initial_codes + textfiles[C])

    cost_result = compute_decoding_performance_from_state(original_state, p_vals, MC_budget)

    logical_error_rates = cost_result['logical_error_rates']
    runtimes = cost_result['runtimes']
    stderrs = cost_result['stderrs']

    original_state_edge_list = parse_edgelist(original_state)

    # Save the original state and results
    original_state_edge_list = np.row_stack(original_state_edge_list, dtype=np.uint8)
    logical_error_rates = np.row_stack(logical_error_rates, dtype=np.float64)

    with h5py.File(output_file, "a") as f:
        grp = f.require_group(grpname[C])
        grp.attrs['MC_budget'] = MC_budget
        
        if 'original_state' in grp:
            del grp['original_state']
        grp.create_dataset("original_state", data=original_state_edge_list, shape=(1, original_state_edge_list.shape[0]))

        if 'physical_error_rates' in grp:
            del grp['physical_error_rates']
        grp.create_dataset("physical_error_rates", data=p_vals)

        if 'logical_error_rates' in grp:
            del grp['logical_error_rates']
        grp.create_dataset("logical_error_rates", data=logical_error_rates)

        if 'logical_error_rates_stderr' in grp:
            del grp['logical_error_rates_stderr']
        grp.create_dataset("logical_error_rates_stderr", data=stderrs)

        if 'runtimes' in grp:
            del grp['runtimes']
        grp.create_dataset("runtimes", data=runtimes)

