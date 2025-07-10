import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix
from sympy import Matrix
from ldpc import mod2

from sympy import Matrix
import numpy as np

def remove_equivalent_logical_operators(logical_ops, stabilizer_basis):
    """
    Remove logical operators equivalent under stabilizer group.
    Returns a set of logically independent operators.
    """
    if len(logical_ops) == 0:
        return np.empty((0, stabilizer_basis.shape[1]), dtype=np.uint8)

    # Stack stabilizers and logicals
    combined = np.vstack([stabilizer_basis, logical_ops])
    row_basis_combined = mod2.row_basis(np.array(combined, dtype=np.uint8))

    rank_logical_ops = mod2.rank(np.array(logical_ops, dtype=np.uint8))
    rank_stabilizers = mod2.rank(np.array(stabilizer_basis, dtype=np.uint8))
    rank_combined = mod2.rank(np.array(combined, dtype=np.uint8))

    print(f"Rank of logical operators: {rank_logical_ops}")
    print(f"Rank of stabilizers: {rank_stabilizers}")
    print(f"Rank of combined basis: {rank_combined}")

    print(f"Row basis combined shape: {row_basis_combined.shape}")
    # print(f"Row basis combined:\n{row_basis_combined.toarray()}")

    logical_ops_reduced = row_basis_combined[stabilizer_basis.shape[0]:]

    return np.array(logical_ops_reduced.toarray(), dtype=np.uint8)

def remove_stabilizers_from_logicals(logical_candidates, stabilizer_generators):
    """
    Remove stabilizers (can be expressed by linear combination of stabilizer generators) from logical operators.
    """

    print(f"stabilizer_generators shape: {stabilizer_generators.shape}")

    row_span = mod2.row_span(np.array(stabilizer_generators, dtype=np.uint8)).toarray().tolist()

    L = []
    for logical in logical_candidates.tolist():
        if logical not in row_span:
            L.append(logical)

    return np.array(L, dtype=np.uint8)

def get_logical_operators(Hx, Hz):
    """
    Get logical operators Lx and Lz from the stabilizer generators Hx and Hz.
    
    Parameters:
        Hx (np.ndarray): Parity check matrix for X-type stabilizers.
        Hz (np.ndarray): Parity check matrix for Z-type stabilizers.
    """

    Lx_candidates = np.array(mod2.kernel(Hz).toarray())
    Lz_candidates = np.array(mod2.kernel(Hx).toarray())

    print(f"Hx shape: {Hx.shape}")
    print(f"Hz shape: {Hz.shape}")

    print(f"Lx_candidates shape: {Lx_candidates.shape}")
    # print(f"Lx_candidates:\n{Lx_candidates}")
    print(f"Lz_candidates shape: {Lz_candidates.shape}")
    # print(f"Lz_candidates:\n{Lz_candidates}")

    Lx_stabilizers_removed = remove_stabilizers_from_logicals(Lx_candidates, Hx)
    Lz_stabilizers_removed = remove_stabilizers_from_logicals(Lz_candidates, Hz)
    print(f"Lx_stabilizers_removed shape: {Lx_stabilizers_removed.shape}")
    # print(f"Lx_stabilizers_removed:\n{Lx_stabilizers_removed}")
    print(f"Lz_stabilizers_removed shape: {Lz_stabilizers_removed.shape}")
    # print(f"Lz_stabilizers_removed:\n{Lz_stabilizers_removed}")

    # Lx = remove_equivalent_logical_operators(Matrix(Lx_stabilizers_removed), Matrix(Hx))
    # Lz = remove_equivalent_logical_operators(Matrix(Lz_stabilizers_removed), Matrix(Hz))

    Lx = np.array(Lx_stabilizers_removed, dtype=np.uint8)
    Lz = np.array(Lz_stabilizers_removed, dtype=np.uint8)
    print(f"Lx:\n{Lx}")
    # print(f"Lz:\n{Lz}")
    return Lx, Lz

if __name__ == "__main__":
    # 7 qubit Steane code example
    Hx = np.array([
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1]
    ], dtype=np.uint8)

    Hz = np.array([
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 1]
    ], dtype=np.uint8)

    # Shor code example

    # Hx = np.array([
    #     [1, 1, 1, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 1, 1, 1, 1, 1, 1]
    # ], dtype=np.uint8)

    # Hz = np.array([
    #     [1, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 1]
    # ], dtype=np.uint8)

    Lx, Lz = get_logical_operators(Hx, Hz)

    print("Lx:\n", Lx)
    print("Lz:\n", Lz)

    # k = Hx.shape[1] - np.linalg.matrix_rank(Hx) - np.linalg.matrix_rank(Hz)
    # Lx = filter_independent_vectors(Lx, k)
    # Lz = filter_independent_vectors(Lz, k)

    # Lx_arr = []
    # Lz_arr = []
    # for i in range(len(Lz)):
    #     Lz_arr.append([int(x) for x in Lz[i]])
    # for i in range(len(Lx)):
    #     Lx_arr.append([int(x) for x in Lx[i]])

    # Lx, Lz = np.array(Lx_arr, dtype=np.uint8), np.array(Lz_arr, dtype=np.uint8)
    # print("Lx:\n", Lx)
    # print("Lz:\n", Lz)
