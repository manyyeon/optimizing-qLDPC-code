import numpy as np

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
def generate_parity_check_matrix(H, L, m, condition_indices):
    n = H * L
    num_checks = L * (H - m + 1)
    H_matrix = np.zeros((num_checks, n), dtype=int)

    check_row = 0
    for i in range(H - m, -1, -1):
        for j in range(L):
            cur_column_idx = (H - i - 1) * L + j
            H_matrix[check_row, (H - i - 1) * L + j] = 1

            # side cell should be applied periodic boundary condition

            condition_offsets = condition_indices
            for dx, dy in condition_offsets:
                target_row = (H - i - 1) - dx
                target_col = (j + dy) % L
                if 0 <= target_row < H:
                    column_idx = target_row * L + target_col
                    H_matrix[check_row, column_idx] = 1

                H_matrix[check_row, column_idx] = 1
            # print index of 1
            # for col in range(H_matrix.shape[1]):
            #     if H_matrix[check_row][col] == 1:
            #         print(f"({check_row}, {col})", end=' ')
            # print()
            check_row += 1

    return H_matrix

def fill_Z_with_stabilizer_shape(input_row, H, L, m, condition_offsets_list, same_shape=False):
    """Evolve the automaton from the input_row using given rule offsets."""
    
    Z = np.zeros((H - m + 1, L), dtype=int)
    Z = np.append(Z, input_row, axis=0)  # append input row at the bottom
    for i in range(H - m, -1, -1):  # evolve upward
        condition_offsets = condition_offsets_list[H - m - i] if same_shape == False else condition_offsets_list[0]  # get the condition offsets for this row
        for j in range(L):
            neighbor_sum = 0
            for dx, dy in condition_offsets:
                neighbor_sum += Z[i + dx, (j + dy) % L]
            Z[i, j] = 1 if neighbor_sum % 2 == 1 else 0  # parity check

    return Z