#!/usr/bin/env python3
"""
Draw Tanner graph figures for the classical [7,4,3] Hamming code.

Outputs:
  1. hamming_743_tanner_graph.{png,svg,pdf}
  2. hamming_743_edge_swap_action.{png,svg,pdf}
"""




def hamming_743_parity_check_matrix() -> np.ndarray:
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=int,
    )














