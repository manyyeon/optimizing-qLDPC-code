import sys
import numpy as np
from scipy.sparse import csr_matrix, hstack, kron, eye, block_diag
from typing import Tuple
from ldpc import mod2

from logical_operators import get_logical_operators_by_pivoting

def get_condition_indices(stabilizer_shape):
    """
    Extracts the list of relative positions (dx, dy) from a m x n binary rule matrix
    that define which neighbor cells contribute to the update rule.

    The (0, *) row is ignored (assumed to represent the current cell),
    and only rows over 1 are used to define condition offsets.
    """
    rows, cols = stabilizer_shape.shape
    print(f"Stabilizer shape:\n{stabilizer_shape}")
    # set (0, 0) for all row 0
    
    relative_positions = [[(i, j) for j in range(-cols//2 + 1, cols//2 + 1)] for i in range(rows)]
    for j in range(cols):
        relative_positions[0][j] = (0, 0)
        
    active_conditions = []               

    for i in range(1, rows):
        for j in range(cols):
            if stabilizer_shape[i][j] == 1:
                active_conditions.append(relative_positions[i][j])

    return active_conditions

# Generate parity-check matrix H
def generate_parity_check_matrix(height, width, m, condition_indices):
    n = height * width
    num_checks = width * (height - m + 1)
    H = np.zeros((num_checks, n), dtype=int)

    check_row = 0
    for i in range(height - m, -1, -1):
        for j in range(width):
            cur_column_idx = (height - i - 1) * width + j
            H[check_row, (height - i - 1) * width + j] = 1

            # side cell should be applied periodic boundary condition

            condition_offsets = condition_indices
            for dx, dy in condition_offsets:
                target_row = (height - i - 1) - dx
                target_col = (j + dy) % width
                if 0 <= target_row < height:
                    column_idx = target_row * width + target_col
                    H[check_row, column_idx] = 1

                H[check_row, column_idx] = 1
            # print index of 1
            # for col in range(H.shape[1]):
            #     if H[check_row][col] == 1:
            #         print(f"({check_row}, {col})", end=' ')
            # print()
            check_row += 1

    return H

def fill_Z_with_stabilizer_shape(input_row, height, width, m, condition_offsets_list, same_shape=False):
    """Evolve the automaton from the input_row using given rule offsets."""

    Z = np.zeros((height - m + 1, width), dtype=int)
    Z = np.append(Z, input_row, axis=0)  # append input row at the bottom
    for i in range(height - m, -1, -1):  # evolve upward
        condition_offsets = condition_offsets_list[height - m - i] if same_shape == False else condition_offsets_list[0]  # get the condition offsets for this row
        for j in range(width):
            neighbor_sum = 0
            for dx, dy in condition_offsets:
                neighbor_sum += Z[i + dx, (j + dy) % width]
            Z[i, j] = 1 if neighbor_sum % 2 == 1 else 0  # parity check

    return Z

def precompute_basis_outputs(height, width, m, condition_indices):
    """Precompute Z for each basis input (single 1 at position i)."""
    Z_basis = []

    for i in range(m-1):
        input_i = np.zeros((1, (m-1) * width), dtype=int)
        input_i = input_i.reshape(m - 1, width)
        input_i[i, 0] = 1
        Z_first = fill_Z_with_stabilizer_shape(input_i, height, width, m, [condition_indices], same_shape=True)
        
        Z_basis.append(Z_first)

        for j in range(1, width):
            Z_j = np.roll(Z_first, j, axis=1)
            Z_basis.append(Z_j)

    """Print the precomputed Z_basis for debugging"""
    # print(f"Precomputed Z_basis for height={height}, width={width}, m={m}:")
    # for idx, z in enumerate(Z_basis):
    #     print(f"Z_basis[{idx}] (index {idx}):\n{z}")

    return Z_basis

def stabilizer_shape_code_matrices(height, width, m, stabilizer_shape) -> Tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """
    Generate the stabilizer shape code matrices.

    Parameters:
        height (int): Height of the code.
        width (int): Width of the code.
        m (int): Number of stabilizers.
        stabilizer_shape (np.ndarray): The shape of the stabilizer.

    Returns:
        list: List of stabilizer shape code matrices.
    """
    n = height * width
    condition_offsets_list = get_condition_indices(stabilizer_shape)
    H = generate_parity_check_matrix(height, width, m, condition_offsets_list)

    codewords = precompute_basis_outputs(height, width, m, condition_offsets_list)
    print(f"Codewords for height={height}, width={width}, m={m}: {len(codewords)} codewords generated.")

    for idx, codeword in enumerate(codewords):
        # reverse the order of rows
        codeword = np.flip(codeword, axis=0)
        codeword = codeword.reshape(1, n)
        codewords[idx] = codeword[0]
    codewords = np.array(codewords, dtype=int)

    Hx = np.zeros((H.shape[0], n), dtype=int)
    Lx_gen_by_al, Lz_gen_by_al = get_logical_operators_by_pivoting(Hx, H)

    # print(f"codewords shape: {codewords.shape}")
    # print(f"Lx shape: {Lx.shape}")
    # print(f"codewords:\n{codewords}")
    # print(f"Lx:\n{Lx}")

    # print(f"Hx @ codewords.T:\n{(np.array(H) @ codewords.T) % 2}")
    # print(f"Hx @ Lx.T:\n{(np.array(H) @ np.array(Lx).T) % 2}")

    # print(f"reduced row echelon form of codewords:\n{mod2.reduced_row_echelon(codewords)[0]}")
    # print(f"reduced row echelon form of Lx:\n{mod2.reduced_row_echelon(Lx)[0]}")

    # print(f"is reduced row echelon form of codewords equal to Lx: {np.array_equal(mod2.reduced_row_echelon(codewords)[0], mod2.reduced_row_echelon(Lx)[0])}")

    Hx = csr_matrix(H)
    Hz = csr_matrix(H)

    Lx = csr_matrix(np.array(codewords, dtype=np.uint8))
    Lz = csr_matrix(np.array(codewords, dtype=np.uint8))

    print(f"Hx shape: {Hx.shape}")
    print(f"Hx: \n{Hx.toarray()}")
    print(f"Hz shape: {Hz.shape}")
    print(f"Hz: \n{Hz.toarray()}")

    print(f"Generated logical operators Lx and Lz by pivoting:")
    print(f"Lx shape: {Lx_gen_by_al.shape}")
    print(f"Lx:\n{Lx_gen_by_al}")
    print(f"Lz shape: {Lz_gen_by_al.shape}")
    print(f"Lz:\n{Lz_gen_by_al}")

    print(f"Hx @ Hz.T should be zero")
    print(f"Hx @ Hz.T:\n{(Hx @ Hz.T).toarray() % 2}")
    
    # Hx @ Lx_gen_by_al.T should be zero
    # print(f"Hx @ Lx_gen_by_al.T:\n{(Hx @ Lx_gen_by_al.T) % 2}")
    # print(f"Hz @ Lz_gen_by_al.T:\n{(Hz @ Lz_gen_by_al.T) % 2}")

    print(f"Generated logical operators Lx and Lz from codewords:")
    print(f"Lx shape: {Lx.shape}")
    print(f"Lx: \n{Lx.toarray()}")
    print(f"Lz shape: {Lz.shape}")
    print(f"Lz: \n{Lz.toarray()}")

    # Hx @ Lx.T should be zero
    # print(f"Hx @ Lx.T:\n{(Hx @ Lx.T.toarray()) % 2}")
    # print(f"Hz @ Lz.T:\n{(Hz @ Lz.T.toarray()) % 2}")


    return Hx, Hz, Lx, Lz
    
if __name__ == "__main__":
    height = 4
    width = 5
    m = 3
    stabilizer_shape = np.array([[0, 1, 0],
                                 [0, 1, 0],
                                 [1, 0, 1]])

    # np.set_printoptions(threshold=sys.maxsize)
    
    Hx, Hz, Lx, Lz = stabilizer_shape_code_matrices(height, width, m, stabilizer_shape)
    print(f"Hx shape: {Hx.shape}")
    print(f"Hz shape: {Hz.shape}")
    print(f"Lx shape: {Lx.shape}")
    print(f"Lz shape: {Lz.shape}")
