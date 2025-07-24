
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimization.experiments_settings import codes, from_edgelist
from optimization.analyze_codes.decoder_performance_from_state import compute_decoding_performance_from_state

import h5py
import argparse
import numpy as np

grpname = codes
p_vals = np.logspace(-2, -1, 20)
MC_budget = int(1e3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=False)
    args = parser.parse_args()

    # Choose the code family
    C = args.C

    fn_data = {}

    with h5py.File('optimization/results/random_exploration_3.hdf5', 'r') as f:
        best_state_edge_list, _ = min(((s, v) for s, v in zip(f[grpname[C]]['states'], f[grpname[C]]['logical_error_rates'])), key=lambda x: x[1])
        index_of_min = np.argmin(f[grpname[C]]['logical_error_rates'])
        print(f"Minimum logical error rate found in state {index_of_min} with value {f[grpname[C]]['logical_error_rates'][index_of_min]}")
        best_state = from_edgelist(best_state_edge_list)

    logical_error_rates = compute_decoding_performance_from_state(best_state, p_vals, MC_budget)

    best_state_edge_list = np.row_stack(best_state_edge_list, dtype=np.uint8)
    logical_error_rates = np.row_stack(logical_error_rates, dtype=np.float64)

    with h5py.File("optimization/results/best_from_exploration.hdf5", "a") as f:
        grp = f.require_group(grpname[C])
        grp.attrs['MC_budget'] = MC_budget

        if "best_state" in grp:
            del grp["best_state"]
        grp.create_dataset("best_state", data=best_state_edge_list, shape=(1, best_state_edge_list.shape[0]))
        
        if "physical_error_rates" in grp:
            del grp["physical_error_rates"]
        grp.create_dataset("physical_error_rates", data=p_vals)

        if "logical_error_rates" in grp:
            del grp["logical_error_rates"]
        grp.create_dataset("logical_error_rates", data=logical_error_rates)
    