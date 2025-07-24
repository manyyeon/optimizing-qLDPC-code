import numpy as np
import numpy.random as npr
import scipy.sparse as sp
from scipy.special import comb

import networkx as nx
import networkx.algorithms.bipartite as bpt
from networkx.algorithms.bipartite import biadjacency_matrix

# import pym4ri
import numba

path_to_initial_codes = 'optimization/initial_codes/'
codes = ['[625,25]', '[1225,65]', '[1600,64]', '[2025,81]']
textfiles = [f"HGP_(3,4)_{code}.txt" for code in codes]

state_space_params = [(15, 20, 60), 
                      (21, 28, 84), 
                      (24, 32, 96), 
                      (27, 36, 108)]

MC_budget = int(1e5)
# noise_levels = [9/32, 8/32, 9/32, 12/32]
noise_levels = [0.01, 8/32, 9/32, 12/32]
# times: 15, 40, 80, 200
p_vals = np.linspace(0.1, 0.5, 15)

def load_tanner_graph(filename: str) -> nx.MultiGraph:
    m, n = np.loadtxt(filename, max_rows=1, dtype=int)
    indices, indptr = np.array([], dtype=int), [0]
    for r in range(m):
        r_ind = np.loadtxt(filename, skiprows=r+1, max_rows=1, dtype=int)
        indices = np.concatenate([indices, np.sort(r_ind)])
        indptr.append(len(r_ind))
    
    H = sp.csr_array((m, n), dtype=int)
    H.data = np.ones_like(indices, dtype=int)
    H.indices = indices
    H.indptr = np.cumsum(indptr)

    return bpt.from_biadjacency_matrix(H, create_using=nx.MultiGraph)


def parse_edgelist(state: nx.MultiGraph) -> np.ndarray:
    return np.array(sorted(state.edges(data=False)), dtype=np.uint8).flatten() # shape: (2*E,)


def from_edgelist(edgelist: np.ndarray) -> nx.MultiGraph:
    diam = lambda arr: np.max(arr) - np.min(arr) + 1
    m, n = np.apply_along_axis(diam, 0, edgelist.reshape(-1,2))

    B = nx.MultiGraph()
    B.add_nodes_from(np.arange(m), bipartite=0)
    B.add_nodes_from(np.arange(m, m+n), bipartite=1)
    B.add_edges_from([tuple(r) for r in edgelist.reshape(-1, 2)])

    return B

def generate_neighbor_highlight(theta: nx.MultiGraph) -> tuple[nx.MultiGraph, tuple]:
    # Copy state
    neighbor = nx.MultiGraph(theta)
    
    # get (multi)edge number from state theta
    E = neighbor.number_of_edges()

    # compute action space size
    A = comb(E, 2, exact=True)
    
    # sample action
    a = npr.choice(A)
    
    # convert to edge indices
    i = np.floor(((2*E - 1) - np.sqrt((2*E-1)**2 - 8*a))//2).astype(int)
    j = (a - E*i + ((i+2)*(i+1))//2)
    
    # apply cross-wiring 
    edge_list = sorted(neighbor.edges(data=False))
    e1, e2 = edge_list[i], edge_list[j]
    (c1, n1), (c2, n2) = e1, e2
    f1, f2 = (c1, n2), (c2, n1)
    neighbor.remove_edges_from([e1, e2])
    neighbor.add_edges_from([f1, f2])
    
    return neighbor, [e1, e2], [f1, f2]


def generate_neighbor(theta: nx.MultiGraph) -> nx.MultiGraph:
    neighbor, *_ = generate_neighbor_highlight(theta)
    return neighbor

@numba.jit(nopython=True)
def gosper_next(c):
    a = c & -c
    b = c + a
    return (((c ^ b) >> 2) // a) | b

def long_gosper_next(c: int) -> int:
    a = c & -c
    b = c + a
    return (((c ^ b) >> 2) // a) | b

@numba.jit(nopython=True)
def _code_distance(H: np.ndarray) -> int:
    _, n = H.shape

    for weight in range(1, n):
        c = (1 << weight) - 1  # Smallest subset of size 'weight'
        while c < (1 << n):
            # Convert 'c' to a binary representation as a NumPy array
            candidate_cw = np.array([(c >> i) & 1 for i in range(n)], dtype=np.float32)

            # Check if it is a codeword
            if not np.any((H @ candidate_cw) % 2):
                return weight

            c = gosper_next(c)  # Get next subset using Gosper's hack


def code_distance(H: sp.csr_array) -> int:
    d = _code_distance((H.astype(np.uint8).todense()&1).astype(np.float32))
    return np.inf if d is None else d

def i2set(i: int, width: int) -> np.ndarray:
    return np.array([(i >> k) & 1 for k in range(width)], dtype=np.int8)

def from_parity_check_matrix(H: sp.csr_array) -> nx.MultiGraph:
    return bpt.from_biadjacency_matrix(H, create_using=nx.MultiGraph)

# def code_dimension(H: sp.csr_array) -> int:
#     _, n = H.shape
#     return n - pym4ri.rank(H.astype(bool).todense())

def sample_LDPC(Lbda: np.poly1d, P: np.poly1d) -> sp.csr_array:
    # Recall graph parameters
    n, m, E = Lbda(1), P(1), P.deriv()(1)
    assert Lbda.deriv()(1) == P.deriv()(1), "Number of edges is inconsistent: Lambda'(1) =/= P'(1)"
    
    # Fill in indptr for csr_array format
    indptr = np.zeros(m+1, dtype=np.int32)
    offset = 1
    for i, P_i in enumerate(P.c[::-1]):
        indptr[offset:offset+P_i] = i
        offset += P_i
    indptr = np.cumsum(indptr)

    # Fill in indices for csr_array format
    indices = np.zeros(E, dtype=np.int32)
    ind_offset, elem_offset = 0, 0
    for i, Lbda_i in enumerate(Lbda.c[::-1]):
        indices[ind_offset:ind_offset+i*Lbda_i] = elem_offset+np.arange(i*Lbda_i)//i
        elem_offset += Lbda_i
        ind_offset += i*Lbda_i
    
    # Apply random edge permutation
    indices = indices[npr.permutation(E)]
    
    # Create sparse biadjacency matrix
    H = sp.csr_array((np.ones(E, dtype=np.int32), indices, indptr), shape=(m, n))
    # Count multiple edges mod 2
    return sp.csr_array(H.todense()&1)

def polys_from_H(H: sp.csr_array) -> tuple[np.poly1d, np.poly1d]:
    return np.poly1d(np.bincount(np.sum(H, axis=0).astype(int))[::-1]), np.poly1d(np.bincount(np.sum(H, axis=1).astype(int))[::-1])



def tanner_graph_to_parity_check_matrix(G: nx.MultiGraph) -> np.ndarray:
    # Get the two bipartite node sets

    c = [n for n, b in G.nodes(data='bipartite') if b == 0]
    v = [n for n, b in G.nodes(data='bipartite') if b == 1]
    
    H = biadjacency_matrix(G, row_order=sorted(c), column_order=sorted(v), dtype=np.bool_)

    return H.toarray()
