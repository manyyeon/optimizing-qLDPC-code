import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix
from sympy import Matrix
from ldpc import mod2

# def gf2_nullspace(A):
#     """Null space of a matrix over GF(2)"""
#     A = Matrix(A.tolist())
#     nullsp = A.nullspace()

#     nullsp_mod2 = np.array([v % 2 for v in nullsp])
#     nullsp_mod2 = np.array([list(map(int, vec)) for vec in nullsp_mod2], dtype=np.uint8)

#     return nullsp_mod2

def is_row_in_span(row, basis):
    """Check if the row is in the span of basis over GF(2)."""
    # Convert everything to sympy Matrix over GF(2)
    A = np.array(basis)
    b = np.array(row)

    augmented = np.vstack([A, b])

    # row rank of A
    rank_A = mod2.rank(A)
    # row rank of augmented matrix
    rank_aug = mod2.rank(augmented) 

    return rank_A == rank_aug

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

    logical_ops_reduced = row_basis_combined[stabilizer_basis.shape[0]:]

    return np.array(logical_ops_reduced.toarray(), dtype=np.uint8)

def remove_stabilizers_from_logicals(logical_candidates, stabilizer_generators):
    """
    Remove stabilizers (can be expressed by linear combination of stabilizer generators) from logical operators.
    """
    L = []
    for logical in logical_candidates:
        if not is_row_in_span(logical, stabilizer_generators):
            L.append(logical % 2)

    return np.array(L, dtype=np.uint8)

def get_logical_operators(Hx, Hz):
    Lx_candidates = np.array(mod2.kernel(Hz).toarray())
    Lz_candidates = np.array(mod2.kernel(Hx).toarray())

    Lx_stabilizers_removed = remove_stabilizers_from_logicals(Lx_candidates, Hx)
    Lz_stabilizers_removed = remove_stabilizers_from_logicals(Lz_candidates, Hz)

    Lx = remove_equivalent_logical_operators(Matrix(Lx_stabilizers_removed), Matrix(Hx))
    Lz = remove_equivalent_logical_operators(Matrix(Lz_stabilizers_removed), Matrix(Hz))

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
