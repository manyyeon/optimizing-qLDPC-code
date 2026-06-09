from collections import defaultdict
import numpy as np
import scipy.sparse as sp


def _column_masks(H):
    """
    Convert each column of a binary parity-check matrix H into an integer bitmask.
    A subset of columns has zero syndrome iff XOR of masks is 0.
    """
    H = sp.csr_matrix(H).astype(np.uint8)
    m, n = H.shape
    Hc = H.tocsc()

    masks = []
    for j in range(n):
        rows = Hc[:, j].indices
        mask = 0
        for r in rows:
            mask ^= (1 << int(r))
        masks.append(mask)

    return masks


def _enumerate_half(masks, max_weight):
    """
    Return counts[weight][syndrome] = number of subsets of this half
    with given weight and syndrome.
    """
    n = len(masks)
    counts = [defaultdict(int) for _ in range(max_weight + 1)]

    subset_weight = [0] * (1 << n)
    subset_syndrome = [0] * (1 << n)

    counts[0][0] = 1

    for subset in range(1, 1 << n):
        lowbit = subset & -subset
        i = lowbit.bit_length() - 1
        prev = subset ^ lowbit

        w = subset_weight[prev] + 1
        if w > max_weight:
            subset_weight[subset] = w
            continue

        syn = subset_syndrome[prev] ^ masks[i]

        subset_weight[subset] = w
        subset_syndrome[subset] = syn
        counts[w][syn] += 1

    return counts


def count_kernel_codewords_by_weight(H, max_weight):
    """
    Count nonzero x in ker(H) by Hamming weight up to max_weight.

    Returns:
        counts_by_weight[w] = number of codewords of weight w.
    """
    masks = _column_masks(H)
    n = len(masks)

    left = masks[: n // 2]
    right = masks[n // 2:]

    left_counts = _enumerate_half(left, max_weight)
    right_counts = _enumerate_half(right, max_weight)

    counts_by_weight = np.zeros(max_weight + 1, dtype=object)

    for w in range(max_weight + 1):
        total = 0
        for wl in range(w + 1):
            wr = w - wl
            for syn, count_l in left_counts[wl].items():
                count_r = right_counts[wr].get(syn, 0)
                total += count_l * count_r
        counts_by_weight[w] = total

    # remove empty codeword
    counts_by_weight[0] -= 1
    return counts_by_weight


def count_parent_low_weight_patterns(H, max_weight):
    """
    Count low-weight undetectable patterns in both H and H^T.
    """
    counts_H = count_kernel_codewords_by_weight(H, max_weight)
    counts_HT = count_kernel_codewords_by_weight(H.T, max_weight)

    # print("Weight | Count in H | Count in H^T | Total")
    # for w in range(1, max_weight + 1):
    #     print(
    #         f"{w:6d} | {counts_H[w]:11d} | {counts_HT[w]:13d} | {counts_H[w] + counts_HT[w]:5d}")

    return {
        "counts_H": counts_H,
        "counts_HT": counts_HT,
        "counts_total": counts_H + counts_HT,
        "total_leq_W": int(np.sum(counts_H[1:]) + np.sum(counts_HT[1:])),
    }
