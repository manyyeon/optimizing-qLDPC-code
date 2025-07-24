import ldpc
import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix
from sympy import Matrix
from ldpc import mod2

import numpy as np

def get_row_echelon(matrix, full=False):
    """
    Converts a binary matrix to row echelon form via Gaussian Elimination

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse
        A binary matrix in either numpy.ndarray format or scipy.sparse
    full: bool, optional
        If set to 'True', Gaussian elimination is only performed on the rows below
        the pivot. If set to 'False' Gaussian eliminatin is performed on rows above
        and below the pivot. 
    
    Returns
    -------
        row_ech_form: numpy.ndarray
            The row echelon form of input matrix
        rank: int
            The rank of the matrix
        transform_matrix: numpy.ndarray
            The transformation matrix such that (transform_matrix@matrix)=row_ech_form
        pivot_cols: list
            List of the indices of pivot num_cols found during Gaussian elimination

    """
    num_rows, num_cols = np.shape(matrix)

    # Take copy of matrix if numpy (why?) and initialise transform matrix to identity
    if isinstance(matrix, np.ndarray):
        the_matrix = np.copy(matrix)
        transform_matrix = np.identity(num_rows).astype(int)
    elif isinstance(matrix, sp.csr.csr_matrix):
        the_matrix = matrix
        transform_matrix = sp.eye(num_rows, dtype="int", format="csr")
    else:
        raise ValueError('Unrecognised matrix type')

    pivot_row = 0
    pivot_cols = []

    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(num_cols):

        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if the_matrix[pivot_row, col] != 1:

            # Find a row with a 1 in this col
            swap_row_index = pivot_row + np.argmax(the_matrix[pivot_row:num_rows, col])

            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if the_matrix[swap_row_index, col] == 1:

                # Swap rows
                the_matrix[[swap_row_index, pivot_row]] = the_matrix[[pivot_row, swap_row_index]]

                # Transformation matrix update to reflect this row swap
                transform_matrix[[swap_row_index, pivot_row]] = transform_matrix[[pivot_row, swap_row_index]]

        # If we have got a pivot, now let's ensure values below that pivot are zeros
        if the_matrix[pivot_row, col]:

            if not full:  
                elimination_range = [k for k in range(pivot_row + 1, num_rows)]
            else:
                elimination_range = [k for k in range(num_rows) if k != pivot_row]

            # Let's zero those values below the pivot by adding our current row to their row
            for j in elimination_range:

                if the_matrix[j, col] != 0 and pivot_row != j:    ### Do we need second condition?

                    the_matrix[j] = (the_matrix[j] + the_matrix[pivot_row]) % 2

                    # Update transformation matrix to reflect this op
                    transform_matrix[j] = (transform_matrix[j] + transform_matrix[pivot_row]) % 2

            pivot_row += 1
            pivot_cols.append(col)

        # Exit loop once there are no more rows to search
        if pivot_row >= num_rows:
            break

    # The rank is equal to the maximum pivot index
    matrix_rank = pivot_row
    row_ech_matrix = the_matrix

    return [row_ech_matrix, matrix_rank, transform_matrix, pivot_cols]

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
    # print(f"Lx:\n{Lx}")
    # print(f"Lz:\n{Lz}")
    return Lx, Lz

def get_logical_operators_by_pivoting(Hx, Hz):
    """
    Get logical operators Lx and Lz from the stabilizer generators Hx and Hz using pivoting.
    
    Parameters:
        Hx (np.ndarray): Parity check matrix for X-type stabilizers.
        Hz (np.ndarray): Parity check matrix for Z-type stabilizers.
    """
    # get X-type logical operators
    # \in ker{Hz} AND \notin Im(Hx)
    ker_Hz = mod2.nullspace(Hz) # compute the kernel basis of Hz
    im_Hx = mod2.row_basis(Hx) # compute the image basis of Hx

    # print(f"ker_Hz shape: {ker_Hz.shape}")
    # print(f"ker_Hz:\n{ker_Hz.toarray()}")
    # print(f"im_Hx shape: {im_Hx.shape}")
    # print(f"im_Hx:\n{im_Hx.toarray()}")

    log_stack = np.vstack([im_Hx.toarray(), ker_Hz.toarray()])

    # print(f"log_stack shape: {log_stack.shape}")
    # print(f"log_stack.T:\n{log_stack.T}")

    row_echelon = get_row_echelon(log_stack.T)

    # print(f"Row echelon form:\n{row_echelon[0]}")
    # print(f"rank: {row_echelon[1]}")
    # print(f"transformations: {row_echelon[2]}")
    # print(f"Row echelon pivot columns: {row_echelon[3]}")
    
    pivot_col_indices = row_echelon[3]

    # print(f"Pivot columns: {pivot_col_indices}")
    log_x_operator_indices = [i for i in range(im_Hx.shape[0], log_stack.shape[0]) if i in pivot_col_indices]
    log_x_ops = log_stack[log_x_operator_indices]

    Lx = np.array(log_x_ops, dtype=np.uint8)

    # print(f"-------------\nFinding logical Z operators")

    # get Z-type logical operators
    # \in ker{Hx} AND \notin Im(Hz)
    ker_Hx = mod2.nullspace(Hx) # compute the kernel basis of Hx
    im_Hz = mod2.row_basis(Hz) # compute the image basis of Hz

    # print(f"ker_Hx shape: {ker_Hx.shape}")
    # print(f"ker_Hx:\n{ker_Hx.toarray()}")
    # print(f"im_Hz shape: {im_Hz.shape}")
    # print(f"im_Hz:\n{im_Hz.toarray()}")

    log_stack = np.vstack([im_Hz.toarray(), ker_Hx.toarray()])

    # print(f"log_stack shape: {log_stack.shape}")
    # print(f"log_stack:\n{log_stack}")

    # row echelon of log_stack.T
    row_echelon = get_row_echelon(log_stack.T)
    pivot_col_indices = row_echelon[3] # pivot column indices

    # print(f"Pivot column indices of log_stack.T: {pivot_col_indices}")

    # log_z_operator_indices = []
    # # print(f"find Z-type logical operators in log_stack.T columns index: {im_Hz.shape[0]} to {log_stack.shape[0]}")
    #  # iterate over the rows of log_stack but exclude the first im_Hz.shape[0] rows (which are the image of Hz)
    # for i in range(im_Hz.shape[0], log_stack):
    #     if i in pivot_col_indices:
    #         # print(f"Row {i} is a logical operator")
    #         log_z_operator_indices.append(i)

    log_z_operator_indices = [i for i in range(im_Hz.shape[0], log_stack.shape[0]) if i in pivot_col_indices]

    log_z_ops = log_stack[log_z_operator_indices]

    Lz = np.array(log_z_ops, dtype=np.uint8)

    print(f"Hx, Hz, Lx, Lz: {Hx.shape}, {Hz.shape}, {Lx.shape}, {Lz.shape}")

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

    Lx, Lz = get_logical_operators_by_pivoting(Hx, Hz)

    # print("Lx:\n", Lx)
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
