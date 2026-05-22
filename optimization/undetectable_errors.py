from itertools import combinations
from typing import Iterable, Optional, Union

import numpy as np
from ldpc import mod2
from scipy.sparse import csr_matrix, issparse


def _as_binary_array(matrix) -> np.ndarray:
    if issparse(matrix):
        return matrix.toarray().astype(np.uint8) % 2
    return np.asarray(matrix, dtype=np.uint8) % 2


def _as_binary_csr(matrix) -> csr_matrix:
    if issparse(matrix):
        out = csr_matrix(matrix, dtype=np.uint8).copy()
    else:
        out = csr_matrix(np.asarray(matrix, dtype=np.uint8) % 2)
    out.data %= 2
    out.eliminate_zeros()
    return out


def _target_weights(distance: int, offsets: Iterable[int]) -> list[int]:
    if not np.isfinite(distance):
        raise ValueError("distance must be finite when offsets are used")
    return [int(distance) + int(offset) for offset in offsets]


def count_span_words_by_weight(
    basis,
    weights: Iterable[int],
    max_comb_order: Optional[int] = None,
    return_words: bool = False,
) -> Union[dict[int, int], tuple[dict[int, int], dict[int, list[np.ndarray]]]]:
    """
    Count nonzero vectors in the binary span of ``basis`` by Hamming weight.

    This enumerates linear combinations of basis rows. If ``max_comb_order`` is
    None, the count is exact over the full span. If it is smaller than the basis
    rank, the result is only for combinations up to that many basis vectors.
    """
    basis_arr = _as_binary_array(basis)
    if basis_arr.ndim == 1:
        basis_arr = basis_arr.reshape(1, -1)

    weights = [int(w) for w in weights]
    counts = {w: 0 for w in weights}
    words = {w: [] for w in weights}

    num_basis = basis_arr.shape[0]
    if num_basis == 0:
        return (counts, words) if return_words else counts

    if max_comb_order is None:
        max_comb_order = num_basis
    max_comb_order = min(int(max_comb_order), num_basis)

    target_set = set(weights)
    for r in range(1, max_comb_order + 1):
        for idxs in combinations(range(num_basis), r):
            vec = np.bitwise_xor.reduce(basis_arr[list(idxs)], axis=0).astype(np.uint8)
            weight = int(vec.sum())
            if weight in target_set:
                counts[weight] += 1
                if return_words:
                    words[weight].append(vec.copy())

    return (counts, words) if return_words else counts


def count_classical_undetectable_errors(
    H,
    distance: int,
    offsets: Iterable[int] = (0, 1, 2),
    max_comb_order: Optional[int] = None,
    return_words: bool = False,
) -> Union[dict[int, int], tuple[dict[int, int], dict[int, list[np.ndarray]]]]:
    """
    Count classical undetectable errors of weights d, d+1, d+2 by default.

    For a classical parity-check matrix H, undetectable errors are exactly the
    nonzero vectors in ker(H). This function counts kernel vectors at the target
    Hamming weights by enumerating combinations of a nullspace basis.
    """
    weights = _target_weights(distance, offsets)
    kernel_basis = mod2.nullspace(_as_binary_csr(H)).toarray().astype(np.uint8)
    return count_span_words_by_weight(
        kernel_basis,
        weights=weights,
        max_comb_order=max_comb_order,
        return_words=return_words,
    )


def sample_weight_undetectable_errors(
    H,
    weight: int,
    num_samples: int = 100,
    seed: Optional[int] = None,
    return_supports: bool = False,
):
    """
    Estimate how often random weight-w errors are undetectable for H.

    An error is undetectable when its syndrome is zero, i.e. H @ e = 0 mod 2.
    Samples are drawn independently and uniformly from all weight-w supports.
    """
    H_csr = _as_binary_csr(H)
    n = H_csr.shape[1]
    weight = int(weight)

    if weight < 0 or weight > n:
        raise ValueError(f"weight must satisfy 0 <= weight <= {n}")

    rng = np.random.default_rng(seed)
    undetectable_count = 0
    undetectable_supports = []

    for _ in range(int(num_samples)):
        support = rng.choice(n, size=weight, replace=False)
        syndrome = np.asarray(H_csr[:, support].sum(axis=1)).ravel() % 2
        if not np.any(syndrome):
            undetectable_count += 1
            if return_supports:
                undetectable_supports.append(np.sort(support).astype(int))

    result = {
        "weight": weight,
        "num_samples": int(num_samples),
        "undetectable_count": undetectable_count,
        "undetectable_fraction": undetectable_count / int(num_samples),
    }
    if return_supports:
        result["undetectable_supports"] = undetectable_supports
    return result


def sample_weights_undetectable_errors(
    H,
    weights: Iterable[int],
    num_samples: int = 100,
    seed: Optional[int] = None,
):
    """
    Run ``sample_weight_undetectable_errors`` for several target weights.
    """
    rng = np.random.default_rng(seed)
    return {
        int(weight): sample_weight_undetectable_errors(
            H,
            weight=int(weight),
            num_samples=num_samples,
            seed=int(rng.integers(0, np.iinfo(np.uint32).max)),
        )
        for weight in weights
    }


def count_css_logical_basis_representatives(
    Hx,
    Hz,
    distance: int,
    offsets: Iterable[int] = (0, 1, 2),
    max_comb_order: Optional[int] = None,
    return_words: bool = False,
):
    """
    Count weights in the span of one chosen CSS logical-operator basis.

    X-type logical representatives come from ker(Hz) / rowspace(Hx), and Z-type
    representatives come from ker(Hx) / rowspace(Hz). This counts combinations
    of the basis returned by get_logical_operators_by_pivoting; it does not
    count every stabilizer-equivalent physical representative.
    """
    from logical_operators import get_logical_operators_by_pivoting

    weights = _target_weights(distance, offsets)
    Lx, Lz = get_logical_operators_by_pivoting(
        _as_binary_array(Hx),
        _as_binary_array(Hz),
    )

    x_result = count_span_words_by_weight(
        Lx,
        weights=weights,
        max_comb_order=max_comb_order,
        return_words=return_words,
    )
    z_result = count_span_words_by_weight(
        Lz,
        weights=weights,
        max_comb_order=max_comb_order,
        return_words=return_words,
    )

    return {"X": x_result, "Z": z_result}
