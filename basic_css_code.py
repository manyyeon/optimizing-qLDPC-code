import numpy as np
from scipy.sparse import csr_matrix, hstack, kron, eye, block_diag
from typing import Tuple
from logical_operators import get_logical_operators

def repetition_code(distance: int) -> csr_matrix:
    row_ind, col_ind = zip(
        *((i, j) for i in range(distance) for j in (i, (i + 1) % distance))
    )
    data = np.ones(2 * distance, dtype=np.uint8)
    return csr_matrix((data, (row_ind, col_ind)))

def toric_code_matrices(
    distance: int,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """Check matrices of a toric code on an unrotated lattice"""
    H = repetition_code(distance=distance)
    # print(f"Toric code distance: {distance}")
    # print(f"Stabilizer matrix H:\n{H.toarray()}")
    assert H.shape[1] == H.shape[0] == distance
    e = eye(distance)

    Hx = csr_matrix(hstack([kron(H, e), kron(e, H.T)], dtype=np.uint8))
    Hz = csr_matrix(hstack([kron(e, H), kron(H.T, e)], dtype=np.uint8))

    L0 = csr_matrix(([1], ([0], [0])), shape=(1, distance), dtype=np.uint8)
    L1 = csr_matrix(np.ones((1, distance), dtype=np.uint8))

    Lx = csr_matrix(block_diag([kron(L0, L1), kron(L1, L0)])) # logical operators (X) - first line of vertical, horizontal - anti-commute with Hz - lines across lattice
    Lz = csr_matrix(block_diag([kron(L1, L0), kron(L0, L1)])) # logical operators (Z) - first line of horizontal, vertical - anti-commute with Hx - lines on the edge of the lattice

    for m in (Hx, Hz, Lx, Lz):
        m.data = m.data % 2
        m.sort_indices()
        m.eliminate_zeros()

    return Hx, Hz, Lx, Lz

def toric_code_matrices_with_logical_operators_by_algorithm(
    distance: int,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """Check matrices of a toric code on an unrotated lattice"""
    H = repetition_code(distance=distance)
    # print(f"Toric code distance: {distance}")
    # print(f"Stabilizer matrix H:\n{H.toarray()}")
    assert H.shape[1] == H.shape[0] == distance
    e = eye(distance)

    Hx = csr_matrix(hstack([kron(H, e), kron(e, H.T)], dtype=np.uint8))
    Hz = csr_matrix(hstack([kron(e, H), kron(H.T, e)], dtype=np.uint8))

    Lx, Lz = get_logical_operators(np.array(Hx.toarray())[:-1, :], np.array(Hz.toarray())[:-1, :])

    Lx = csr_matrix(Lx, dtype=np.uint8)
    Lz = csr_matrix(Lz, dtype=np.uint8)

    return Hx, Hz, Lx, Lz
    

def shor_code_matrices():
    # 9 qubit Shor code
    Hx = np.array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1]
    ], dtype=np.uint8)

    Hz = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1]
    ], dtype=np.uint8)

    Lx, Lz = get_logical_operators(Hx, Hz)

    # Convert logical operators to sparse matrices
    Hx = csr_matrix(Hx, dtype=np.uint8)
    Hz = csr_matrix(Hz, dtype=np.uint8)
    Lx = csr_matrix(Lx, dtype=np.uint8)
    Lz = csr_matrix(Lz, dtype=np.uint8)

    return Hx, Hz, Lx, Lz

if __name__ == "__main__":

    distance = 3  # Example distance
    # Hx, Hz, Lx, Lz = toric_code_matrices(distance)
    # Hx, Hz, Lx, Lz = toric_code_matrices_with_logical_operators_by_algorithm(distance)
    Hx, Hz, Lx, Lz = shor_code_matrices()

    print(f"Hx:\n{Hx.toarray()}")
    print(f"Hz:\n{Hz.toarray()}")
    print(f"Lx:\n{Lx.toarray()}")
    print(f"Lz:\n{Lz.toarray()}")

