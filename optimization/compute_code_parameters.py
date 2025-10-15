from typing import Tuple
import ldpc
import ldpc.code_util
from scipy.sparse import csr_matrix
from optimization.experiments_settings import tanner_graph_to_parity_check_matrix
import networkx as nx
from basic_css_code import construct_HGP_code
import numpy as np


def compute_code_parameters(state: nx.MultiGraph) -> Tuple[int, int, int]:
    """
    Compute the code parameters [n, k, d] from the given Tanner graph state.
    Parameters:
        state (nx.MultiGraph): The Tanner graph representing the code.
    Returns:
        n (int): Length of the code (number of variable nodes).
        k (int): Dimension of the code (number of logical qubits).
        d (int): Minimum distance of the code.
    """

    # Convert the Tanner graph to a parity-check matrix (ndarray format)
    H = tanner_graph_to_parity_check_matrix(state)
    H = csr_matrix(H, dtype=np.uint8)
    # Hx, Hz = construct_HGP_code(H)

    n, k, d = ldpc.code_util.compute_code_parameters(H)
    print(f"Code parameters: n={n}, k={k}, d={d}")

    return n, k, d