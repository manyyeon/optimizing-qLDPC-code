from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    from ldpc.bposd_decoder import BpOsdDecoder
except ModuleNotFoundError:  # Allows exact/landscape-only runs without ldpc installed.
    BpOsdDecoder = None


Term = tuple[int, int]
LogicalType = Literal["X", "Z"]
SearchStrategy = Literal["targeted", "random"]


# =============================================================================
# GF(2) linear algebra
# =============================================================================


def gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    a = np.asarray(matrix, dtype=np.uint8).copy() % 2
    m, n = a.shape
    pivots: list[int] = []
    pivot_row = 0

    for col in range(n):
        if pivot_row >= m:
            break
        candidates = np.flatnonzero(a[pivot_row:, col])
        if candidates.size == 0:
            continue

        row = pivot_row + int(candidates[0])
        if row != pivot_row:
            a[[pivot_row, row]] = a[[row, pivot_row]]

        for r in range(m):
            if r != pivot_row and a[r, col]:
                a[r] ^= a[pivot_row]

        pivots.append(col)
        pivot_row += 1

    return a, pivots


def gf2_rank(matrix: np.ndarray) -> int:
    return len(gf2_rref(matrix)[1])


def gf2_row_basis(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.uint8) % 2
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[0] == 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.uint8)
    rref, pivots = gf2_rref(matrix)
    return rref[: len(pivots)].copy()


def gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    a = np.asarray(matrix, dtype=np.uint8) % 2
    rref, pivots = gf2_rref(a)
    n = a.shape[1]
    pivot_set = set(pivots)
    free_cols = [col for col in range(n) if col not in pivot_set]

    basis: list[np.ndarray] = []
    for free_col in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[free_col] = 1
        for row, pivot_col in enumerate(pivots):
            vec[pivot_col] = rref[row, free_col]
        basis.append(vec)

    if not basis:
        return np.zeros((0, n), dtype=np.uint8)
    return np.asarray(basis, dtype=np.uint8)


def quotient_basis(
    vector_space_basis: np.ndarray,
    subspace_basis: np.ndarray,
) -> np.ndarray:
    """Return rows extending subspace_basis to vector_space_basis over GF(2)."""
    vector_space_basis = np.asarray(vector_space_basis, dtype=np.uint8) % 2
    subspace_basis = np.asarray(subspace_basis, dtype=np.uint8) % 2
    ambient_dim = vector_space_basis.shape[1]

    span = gf2_row_basis(subspace_basis)
    current_rank = span.shape[0]
    representatives: list[np.ndarray] = []

    for vec in vector_space_basis:
        trial = (
            np.vstack((span, vec))
            if span.size
            else vec.reshape(1, ambient_dim)
        )
        new_rank = gf2_rank(trial)
        if new_rank > current_rank:
            representatives.append(vec.copy())
            span = gf2_row_basis(trial)
            current_rank = new_rank

    if not representatives:
        return np.zeros((0, ambient_dim), dtype=np.uint8)
    return np.asarray(representatives, dtype=np.uint8)


def css_logical_bases(
    hx: np.ndarray,
    hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return quotient-basis representatives.

    logical X = ker(Hz) / row(Hx)
    logical Z = ker(Hx) / row(Hz)
    """
    logical_x = quotient_basis(gf2_nullspace(hz), gf2_row_basis(hx))
    logical_z = quotient_basis(gf2_nullspace(hx), gf2_row_basis(hz))
    return logical_x, logical_z


# =============================================================================
# BB construction
# =============================================================================


@dataclass(frozen=True)
class BBState:
    ell: int
    m: int
    a_terms: tuple[Term, ...]
    b_terms: tuple[Term, ...]

    def canonical(self) -> "BBState":
        a = tuple(sorted((u % self.ell, v % self.m) for u, v in self.a_terms))
        b = tuple(sorted((u % self.ell, v % self.m) for u, v in self.b_terms))
        return BBState(self.ell, self.m, a, b)


def allowed_pure_axis_terms(ell: int, m: int) -> tuple[Term, ...]:
    """Restricted BB terms: 1, x^a, or y^b."""
    return tuple(
        [(0, 0)]
        + [(a, 0) for a in range(1, ell)]
        + [(0, b) for b in range(1, m)]
    )


def monomial_matrix(ell: int, m: int, term: Term) -> np.ndarray:
    """Binary permutation matrix for x^u y^v."""
    u, v = term
    u %= ell
    v %= m
    size = ell * m
    matrix = np.zeros((size, size), dtype=np.uint8)

    for i in range(ell):
        for j in range(m):
            row = i * m + j
            col = ((i + u) % ell) * m + ((j + v) % m)
            matrix[row, col] = 1

    return matrix


def polynomial_matrix(
    ell: int,
    m: int,
    terms: Iterable[Term],
) -> np.ndarray:
    size = ell * m
    matrix = np.zeros((size, size), dtype=np.uint8)
    for term in terms:
        matrix ^= monomial_matrix(ell, m, term)
    return matrix


def build_bb_checks(
    state: BBState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = state.canonical()
    if len(state.a_terms) != 3 or len(state.b_terms) != 3:
        raise ValueError("This script expects exactly three terms in A and B.")
    if len(set(state.a_terms)) != len(state.a_terms):
        raise ValueError("A contains duplicate terms, which cancel over GF(2).")
    if len(set(state.b_terms)) != len(state.b_terms):
        raise ValueError("B contains duplicate terms, which cancel over GF(2).")

    a = polynomial_matrix(state.ell, state.m, state.a_terms)
    b = polynomial_matrix(state.ell, state.m, state.b_terms)
    hx = np.hstack((a, b)).astype(np.uint8)
    hz = np.hstack((b.T, a.T)).astype(np.uint8)

    if np.any((hx @ hz.T) % 2):
        raise ValueError("CSS commutation failed: Hx Hz^T != 0.")

    return a, b, hx, hz


def css_tanner_is_connected(hx: np.ndarray, hz: np.ndarray) -> bool:
    """Connectivity of the Tanner graph with X checks, Z checks, and qubits."""
    check_matrix = csr_matrix(np.vstack((hx, hz)), dtype=np.uint8)
    adjacency = bmat(
        [[None, check_matrix], [check_matrix.T, None]],
        format="csr",
    )
    n_components, _ = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    return n_components == 1


def compute_k(hx: np.ndarray, hz: np.ndarray) -> tuple[int, int, int]:
    n = hx.shape[1]
    rank_hx = gf2_rank(hx)
    rank_hz = gf2_rank(hz)
    return n - rank_hx - rank_hz, rank_hx, rank_hz


# =============================================================================
# Exact CSS distance by MILP (final certification only)
# =============================================================================


@dataclass
class DistanceResult:
    weight: int
    logical: np.ndarray
    certified: bool
    status: int
    message: str
    lower_bound: float | None
    mip_gap: float | None


def exact_minimum_logical_milp(
    detecting_check: np.ndarray,
    opposite_logical_basis: np.ndarray,
    *,
    time_limit: float = 60.0,
) -> DistanceResult:
    """Find a minimum logical, returning an incumbent even on timeout.

    ``certified`` is true only when HiGHS proves optimality.  Otherwise the
    returned weight is merely an upper bound on the true distance.
    """
    h = np.asarray(detecting_check, dtype=np.uint8) % 2
    logical_basis = np.asarray(opposite_logical_basis, dtype=np.uint8) % 2
    m_checks, n = h.shape
    k = logical_basis.shape[0]
    if k == 0:
        raise ValueError("The code encodes no logical qubits.")

    z_start = 0
    t_start = n
    q_start = n + m_checks
    y_start = n + m_checks + k
    total_vars = n + m_checks + 2 * k

    objective = np.zeros(total_vars, dtype=float)
    objective[z_start:t_start] = 1.0
    integrality = np.ones(total_vars, dtype=int)
    lower = np.zeros(total_vars, dtype=float)
    upper = np.zeros(total_vars, dtype=float)
    upper[z_start:t_start] = 1.0
    upper[t_start:q_start] = h.sum(axis=1) // 2
    upper[q_start:y_start] = logical_basis.sum(axis=1) // 2
    upper[y_start:] = 1.0

    equality = np.zeros((m_checks + k, total_vars), dtype=float)
    rhs = np.zeros(m_checks + k, dtype=float)
    equality[:m_checks, z_start:t_start] = h
    equality[:m_checks, t_start:q_start] = -2.0 * np.eye(m_checks)
    equality[m_checks:, z_start:t_start] = logical_basis
    equality[m_checks:, q_start:y_start] = -2.0 * np.eye(k)
    equality[m_checks:, y_start:] = -1.0 * np.eye(k)

    at_least_one = np.zeros((1, total_vars), dtype=float)
    at_least_one[0, y_start:] = 1.0

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=[
            LinearConstraint(csr_matrix(equality), rhs, rhs),
            LinearConstraint(csr_matrix(at_least_one), np.array([1.0]), np.array([np.inf])),
        ],
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": 0.0,
            "presolve": True,
            "disp": False,
        },
    )

    if result.x is None:
        raise RuntimeError(
            "MILP found no incumbent logical operator. "
            f"status={result.status}, message={result.message}"
        )

    logical = np.rint(result.x[:n]).astype(np.uint8)
    if np.any((h @ logical) % 2):
        raise RuntimeError("MILP incumbent has nonzero syndrome.")
    if not np.any((logical_basis @ logical) % 2):
        raise RuntimeError("MILP incumbent is logically trivial.")

    return DistanceResult(
        weight=int(logical.sum()),
        logical=logical,
        certified=(result.status == 0),
        status=int(result.status),
        message=str(result.message),
        lower_bound=(None if getattr(result, "mip_dual_bound", None) is None else float(result.mip_dual_bound)),
        mip_gap=(None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)),
    )


@dataclass
class BBEvaluation:
    state: BBState
    n: int
    k: int
    rank_hx: int
    rank_hz: int
    connected: bool
    d: int
    d_x: int
    d_z: int
    d_certified: bool
    d_x_certified: bool
    d_z_certified: bool
    min_x_logical: np.ndarray
    min_z_logical: np.ndarray
    hx: np.ndarray
    hz: np.ndarray


def evaluate_bb_state(
    state: BBState,
    *,
    distance_time_limit: float = 60.0,
) -> BBEvaluation:
    """Expensive exact/incumbent evaluation; use only for final certification."""
    _, _, hx, hz = build_bb_checks(state)
    k, rank_hx, rank_hz = compute_k(hx, hz)
    connected = css_tanner_is_connected(hx, hz)
    if k <= 0:
        raise ValueError(f"Candidate encodes no logical qubits: k={k}.")

    logical_x_basis, logical_z_basis = css_logical_bases(hx, hz)
    if logical_x_basis.shape[0] != k or logical_z_basis.shape[0] != k:
        raise RuntimeError(
            "Logical-basis dimension disagrees with CSS rank formula: "
            f"k={k}, dim(Lx)={logical_x_basis.shape[0]}, dim(Lz)={logical_z_basis.shape[0]}."
        )

    z_result = exact_minimum_logical_milp(
        hx, logical_x_basis, time_limit=distance_time_limit
    )
    x_result = exact_minimum_logical_milp(
        hz, logical_z_basis, time_limit=distance_time_limit
    )
    d = min(x_result.weight, z_result.weight)
    # The CSS distance is certified only when both X and Z optimizations
    # are certified; an uncertified side could still contain a lighter logical.
    d_certified = x_result.certified and z_result.certified
    return BBEvaluation(
        state=state.canonical(), n=hx.shape[1], k=k,
        rank_hx=rank_hx, rank_hz=rank_hz, connected=connected,
        d=d, d_x=x_result.weight, d_z=z_result.weight,
        d_certified=d_certified,
        d_x_certified=x_result.certified,
        d_z_certified=z_result.certified,
        min_x_logical=x_result.logical,
        min_z_logical=z_result.logical,
        hx=hx, hz=hz,
    )


# =============================================================================
# BP-OSD search evaluation, sampled spectrum, and targets
# =============================================================================


@dataclass
class LogicalTarget:
    logical_type: LogicalType
    vector: np.ndarray
    weight: int
    source: str


@dataclass
class SpectrumAnalysis:
    score: float
    components: dict[int, int]
    components_x: dict[int, int]
    components_z: dict[int, int]
    min_weight: int
    max_weight: int
    targets: list[LogicalTarget]
    x_decoder_failures: int
    z_decoder_failures: int
    num_probes: int


@dataclass
class SearchEvaluation:
    state: BBState
    n: int
    k: int
    rank_hx: int
    rank_hz: int
    connected: bool
    d_est: int
    d_x_est: int
    d_z_est: int
    min_x_logical: np.ndarray
    min_z_logical: np.ndarray
    hx: np.ndarray
    hz: np.ndarray


def generate_logical_class_probes(k: int, num_probes: int, rng: np.random.Generator) -> np.ndarray:
    """Generate distinct nonzero coefficient vectors in F_2^k."""
    total_classes = (1 << k) - 1
    count = total_classes if num_probes <= 0 else min(num_probes, total_classes)
    if count == total_classes:
        labels = np.arange(1, total_classes + 1, dtype=np.uint64)
    else:
        labels = rng.choice(np.arange(1, total_classes + 1, dtype=np.uint64), size=count, replace=False)
    bit_positions = np.arange(k, dtype=np.uint64)

    return ((labels[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)


def make_bposd_coset_decoder(
    parity_check: np.ndarray,
    *, error_rate: float = 0.05, max_iter: int = 1000, osd_order: int = 2,
):
    if BpOsdDecoder is None:
        raise ModuleNotFoundError("The 'ldpc' package is required for BP-OSD search evaluation.")
    return BpOsdDecoder(
        csr_matrix(parity_check, dtype=np.uint8),
        error_rate=error_rate, max_iter=max_iter,
        bp_method="ms", ms_scaling_factor=0.625,
        osd_method="osd_cs", osd_order=osd_order,
    )


def sample_logical_class_leaders(
    logical_basis: np.ndarray,
    stabilizer_generator: np.ndarray,
    detecting_check: np.ndarray,
    probe_coefficients: np.ndarray,
    *, decoder_error_rate: float, decoder_max_iter: int, decoder_osd_order: int,
) -> dict:
    logical_basis = np.asarray(logical_basis, dtype=np.uint8) % 2
    print(f"logical basis: {logical_basis}")
    stabilizer_generator = np.asarray(stabilizer_generator, dtype=np.uint8) % 2
    print(f"stabilizer generator: {stabilizer_generator}")
    detecting_check = np.asarray(detecting_check, dtype=np.uint8) % 2
    print(f"detecting check: {detecting_check}")
    coset_check = gf2_nullspace(stabilizer_generator)
    print(f"coset check: {coset_check}")

    decoder = make_bposd_coset_decoder(
        coset_check, error_rate=decoder_error_rate,
        max_iter=decoder_max_iter, osd_order=decoder_osd_order,
    )

    counts: defaultdict[int, int] = defaultdict(int)
    results: list[dict] = []
    failed = 0

    print(f"probe coefficients: {probe_coefficients}")

    for coefficients in probe_coefficients:
        representative = (coefficients @ logical_basis).astype(np.uint8) % 2
        syndrome = (coset_check @ representative).astype(np.uint8) % 2
        leader = np.asarray(decoder.decode(syndrome), dtype=np.uint8).reshape(-1) % 2

        if not np.array_equal((coset_check @ leader) % 2, syndrome):
            failed += 1
            print(f"coset check @ leader mod 2 != syndrome")
            continue
            
        if np.any((detecting_check @ leader) % 2):
            failed += 1
            print(f"detecting check failed")
            continue
        
        weight = int(leader.sum())
        counts[weight] += 1
        results.append({"weight": weight, "logical": leader.copy()})

        print(f"counts: {counts}")

    return {"counts": dict(counts), "results": results,
            "num_successful": len(results), "num_failed": failed}


def _deduplicate_targets(targets: Sequence[LogicalTarget]) -> list[LogicalTarget]:
    best: dict[tuple[str, tuple[int, ...]], LogicalTarget] = {}
    for target in targets:
        key = (target.logical_type, tuple(np.flatnonzero(target.vector)))
        old = best.get(key)
        if old is None or (target.weight, target.source) < (old.weight, old.source):
            best[key] = target
    return sorted(best.values(), key=lambda x: (x.weight, x.logical_type, x.source))


def _minimum_result(result: dict, side: str) -> tuple[int, np.ndarray]:
    if not result["results"]:
        raise RuntimeError(f"BP+OSD recovered no valid {side}-logical representatives.")
    
    best = min(result["results"], key=lambda row: int(row["weight"]))
    
    return int(best["weight"]), np.asarray(best["logical"], dtype=np.uint8).copy()


def evaluate_search_state(
    state: BBState,
    probe_coefficients: np.ndarray,
    *, gamma: float, max_weight_offset: int, max_targets_per_type: int,
    decoder_error_rate: float, decoder_max_iter: int, decoder_osd_order: int,
) -> tuple[SearchEvaluation, SpectrumAnalysis]:
    """Evaluate a state for search without solving a MILP.

    The returned d estimates are upper bounds obtained from the sampled logical
    classes and BP+OSD representatives; they are not certified distances.
    """
    _, _, hx, hz = build_bb_checks(state)
    print(f"hx: {hx}, \n hz: {hz}")
    k, rank_hx, rank_hz = compute_k(hx, hz)
    print(f"k: {k}, rank_hx={rank_hx}, rank_hz={rank_hz}")

    if k <= 0:
        raise ValueError(f"Candidate encodes no logical qubits: k={k}.")
    logical_x_basis, logical_z_basis = css_logical_bases(hx, hz)

    if logical_x_basis.shape[0] != k or logical_z_basis.shape[0] != k:
        raise RuntimeError("Logical-basis dimension disagrees with CSS rank formula.")
    
    print(f"probe_coefficients: {probe_coefficients.shape}")
    if probe_coefficients.shape[1] != k:
        raise ValueError(f"Probe width {probe_coefficients.shape[1]} does not match k={k}.")

    z_result = sample_logical_class_leaders(
        logical_z_basis, hz, hx, probe_coefficients,
        decoder_error_rate=decoder_error_rate,
        decoder_max_iter=decoder_max_iter,
        decoder_osd_order=decoder_osd_order,
    )
    
    print(f"z_result: {z_result}")

    x_result = sample_logical_class_leaders(
        logical_x_basis, hx, hz, probe_coefficients,
        decoder_error_rate=decoder_error_rate,
        decoder_max_iter=decoder_max_iter,
        decoder_osd_order=decoder_osd_order,
    )
    print(f"x_result: {x_result}")

    d_z_est, min_z = _minimum_result(z_result, "Z")
    d_x_est, min_x = _minimum_result(x_result, "X")
    d_est = min(d_x_est, d_z_est)
    min_weight = d_est
    max_weight = d_est + max_weight_offset

    score = 0.0
    components_x: dict[int, int] = {}
    components_z: dict[int, int] = {}
    components: dict[int, int] = {}
    for weight in range(min_weight, max_weight + 1):
        cx = int(x_result["counts"].get(weight, 0))
        cz = int(z_result["counts"].get(weight, 0))
        components_x[weight] = cx; components_z[weight] = cz; components[weight] = cx + cz
        score += (gamma ** (weight - d_est)) * (
            cx / x_result["num_successful"] + cz / z_result["num_successful"]
        )

    targets: list[LogicalTarget] = []
    for logical_type, result in (("X", x_result), ("Z", z_result)):
        retained = 0; seen: set[tuple[int, ...]] = set()
        for item in sorted(result["results"], key=lambda row: int(row["weight"])):
            weight = int(item["weight"])
            if weight > max_weight:
                continue
            vector = np.asarray(item["logical"], dtype=np.uint8)
            key = tuple(np.flatnonzero(vector))
            if key in seen:
                continue
            seen.add(key)
            targets.append(LogicalTarget(logical_type, vector.copy(), weight, "BP-OSD"))
            retained += 1
            if retained >= max_targets_per_type:
                break

    evaluation = SearchEvaluation(
        state=state.canonical(), n=hx.shape[1], k=k,
        rank_hx=rank_hx, rank_hz=rank_hz,
        connected=css_tanner_is_connected(hx, hz),
        d_est=d_est, d_x_est=d_x_est, d_z_est=d_z_est,
        min_x_logical=min_x, min_z_logical=min_z, hx=hx, hz=hz,
    )
    spectrum = SpectrumAnalysis(
        score=float(score), components=components,
        components_x=components_x, components_z=components_z,
        min_weight=min_weight, max_weight=max_weight,
        targets=_deduplicate_targets(targets),
        x_decoder_failures=int(x_result["num_failed"]),
        z_decoder_failures=int(z_result["num_failed"]),
        num_probes=int(len(probe_coefficients)),
    )
    return evaluation, spectrum


# =============================================================================
# One-term and paired BB actions
# =============================================================================


@dataclass(frozen=True)
class TermReplacement:
    block: Literal["A", "B"]
    old_term: Term
    new_term: Term


@dataclass(frozen=True)
class BBAction:
    replacements: tuple[TermReplacement, ...]

    @property
    def size(self) -> int:
        return len(self.replacements)

    def label(self) -> str:
        return "; ".join(
            f"{r.block}:{r.old_term}->{r.new_term}" for r in self.replacements
        )


@dataclass
class StructuralProposal:
    action: BBAction
    state: BBState
    broken_target_count: int
    weighted_coverage: float
    total_syndrome_weight: int


def enumerate_single_replacements(state: BBState) -> list[TermReplacement]:
    allowed = allowed_pure_axis_terms(state.ell, state.m)
    replacements: list[TermReplacement] = []

    for block, terms in (("A", state.a_terms), ("B", state.b_terms)):
        current_set = set(terms)
        for old_term in terms:
            for new_term in allowed:
                if new_term == old_term or new_term in current_set:
                    continue
                replacements.append(TermReplacement(block, old_term, new_term))

    return replacements


def apply_action(state: BBState, action: BBAction) -> BBState:
    a_terms = list(state.a_terms)
    b_terms = list(state.b_terms)
    occupied_positions: set[tuple[str, Term]] = set()

    for replacement in action.replacements:
        position = (replacement.block, replacement.old_term)
        if position in occupied_positions:
            raise ValueError("An action cannot replace the same term twice.")
        occupied_positions.add(position)

        terms = a_terms if replacement.block == "A" else b_terms
        try:
            index = terms.index(replacement.old_term)
        except ValueError as exc:
            raise ValueError(f"Old term {replacement.old_term} not present.") from exc
        terms[index] = replacement.new_term

    if len(set(a_terms)) != 3 or len(set(b_terms)) != 3:
        raise ValueError("Action creates duplicate terms in A or B.")

    new_state = replace(state, a_terms=tuple(a_terms), b_terms=tuple(b_terms)).canonical()
    if new_state == state.canonical():
        raise ValueError("Action leaves the canonical state unchanged.")
    return new_state


def enumerate_actions(
    state: BBState,
    *,
    include_pairs: bool,
) -> list[tuple[BBAction, BBState]]:
    """Enumerate unique one-term and simultaneous two-term neighbor states."""
    singles = enumerate_single_replacements(state)
    by_state: dict[BBState, BBAction] = {}

    for replacement in singles:
        action = BBAction((replacement,))
        try:
            new_state = apply_action(state, action)
        except ValueError:
            continue
        by_state.setdefault(new_state, action)

    if include_pairs:
        for first, second in combinations(singles, 2):
            if (first.block, first.old_term) == (second.block, second.old_term):
                continue
            action = BBAction((first, second))
            try:
                new_state = apply_action(state, action)
            except ValueError:
                continue
            old_action = by_state.get(new_state)
            if old_action is None or action.size < old_action.size:
                by_state[new_state] = action

    return [(action, new_state) for new_state, action in by_state.items()]


def target_coverage(
    parent: SearchEvaluation,
    hx_new: np.ndarray,
    hz_new: np.ndarray,
    targets: Sequence[LogicalTarget],
    *,
    gamma: float,
) -> tuple[int, float, int]:
    broken_count = 0
    weighted_coverage = 0.0
    total_syndrome_weight = 0

    for target in targets:
        check_new = hx_new if target.logical_type == "Z" else hz_new
        syndrome_weight = int(((check_new @ target.vector) % 2).sum())
        if syndrome_weight == 0:
            continue

        broken_count += 1
        total_syndrome_weight += syndrome_weight
        exponent = max(0, target.weight - parent.d_est)
        weighted_coverage += gamma**exponent

    return broken_count, float(weighted_coverage), total_syndrome_weight


def select_pair_proposals(
    proposals: list[StructuralProposal],
    *,
    max_pair_proposals: int,
    rng: np.random.Generator,
) -> list[StructuralProposal]:
    singles = [proposal for proposal in proposals if proposal.action.size == 1]
    pairs = [proposal for proposal in proposals if proposal.action.size == 2]

    if max_pair_proposals <= 0 or len(pairs) <= max_pair_proposals:
        return singles + pairs

    pairs.sort(
        key=lambda proposal: (
            -proposal.weighted_coverage,
            -proposal.broken_target_count,
            -proposal.total_syndrome_weight,
            proposal.action.label(),
        )
    )

    top_count = max_pair_proposals // 2
    chosen = pairs[:top_count]
    tail = pairs[top_count:]
    random_count = max_pair_proposals - len(chosen)
    if random_count > 0 and tail:
        indices = rng.choice(
            len(tail),
            size=min(random_count, len(tail)),
            replace=False,
        )
        chosen.extend(tail[int(index)] for index in indices)

    return singles + chosen


def enumerate_structural_proposals(
    parent: SearchEvaluation,
    targets: Sequence[LogicalTarget],
    *,
    strategy: SearchStrategy,
    target_k: int,
    require_connected: bool,
    include_pairs: bool,
    max_pair_proposals: int,
    children_per_parent: int,
    coverage_gamma: float,
    rng: np.random.Generator,
    visited_states: set[BBState],
) -> list[StructuralProposal]:
    proposals: list[StructuralProposal] = []

    for action, state in enumerate_actions(parent.state, include_pairs=include_pairs):
        if state in visited_states:
            continue

        _, _, hx_new, hz_new = build_bb_checks(state)
        k_new, _, _ = compute_k(hx_new, hz_new)
        if k_new != target_k:
            continue
        if require_connected and not css_tanner_is_connected(hx_new, hz_new):
            continue

        broken_count, coverage, syndrome_weight = target_coverage(
            parent,
            hx_new,
            hz_new,
            targets,
            gamma=coverage_gamma,
        )

        if strategy == "targeted" and broken_count == 0:
            continue

        proposals.append(
            StructuralProposal(
                action=action,
                state=state,
                broken_target_count=broken_count,
                weighted_coverage=coverage,
                total_syndrome_weight=syndrome_weight,
            )
        )

    proposals = select_pair_proposals(
        proposals,
        max_pair_proposals=max_pair_proposals,
        rng=rng,
    )

    if strategy == "targeted":
        proposals.sort(
            key=lambda proposal: (
                -proposal.weighted_coverage,
                -proposal.broken_target_count,
                -proposal.total_syndrome_weight,
                proposal.action.size,
                proposal.action.label(),
            )
        )
        if children_per_parent > 0:
            proposals = proposals[:children_per_parent]
    else:
        if children_per_parent > 0 and len(proposals) > children_per_parent:
            indices = rng.choice(
                len(proposals),
                size=children_per_parent,
                replace=False,
            )
            proposals = [proposals[int(index)] for index in indices]
        rng.shuffle(proposals)

    return proposals


# =============================================================================
# Beam search
# =============================================================================


@dataclass
class SearchNode:
    strategy: SearchStrategy
    depth: int
    evaluation: SearchEvaluation
    spectrum: SpectrumAnalysis
    parent_state: BBState | None
    action: BBAction | None
    broken_target_count: int
    weighted_coverage: float
    total_syndrome_weight: int


def state_sort_key(state: BBState) -> tuple:
    return state.a_terms, state.b_terms


def node_rank_key(node: SearchNode, *, use_coverage: bool) -> tuple:
    return (
        -node.evaluation.d_est,
        float(node.spectrum.score),
        -node.weighted_coverage if use_coverage else 0.0,
        -node.broken_target_count if use_coverage else 0,
        -node.total_syndrome_weight if use_coverage else 0,
        node.action.size if node.action is not None else 0,
        state_sort_key(node.evaluation.state),
    )


def run_beam_search(
    initial: SearchEvaluation,
    initial_spectrum: SpectrumAnalysis,
    *, strategy: SearchStrategy, steps: int, beam_width: int,
    children_per_parent: int, include_pairs: bool, max_pair_proposals: int,
    target_k: int, require_connected: bool, max_distance_drop: int,
    probe_coefficients: np.ndarray, score_gamma: float,
    score_max_weight_offset: int, max_targets_per_type: int,
    decoder_error_rate: float, decoder_max_iter: int, decoder_osd_order: int,
    rng: np.random.Generator,
    evaluation_cache: dict[BBState, tuple[SearchEvaluation, SpectrumAnalysis]] | None = None,
) -> tuple[SearchNode, list[SearchNode]]:
    cache = {} if evaluation_cache is None else evaluation_cache
    cache[initial.state] = (initial, initial_spectrum)

    def get_evaluation(state: BBState) -> tuple[SearchEvaluation, SpectrumAnalysis]:
        cached = cache.get(state)
        if cached is None:
            cached = evaluate_search_state(
                state, probe_coefficients,
                gamma=score_gamma,
                max_weight_offset=score_max_weight_offset,
                max_targets_per_type=max_targets_per_type,
                decoder_error_rate=decoder_error_rate,
                decoder_max_iter=decoder_max_iter,
                decoder_osd_order=decoder_osd_order,
            )
            cache[state] = cached
        return cached

    initial_node = SearchNode(
        strategy=strategy, depth=0, evaluation=initial,
        spectrum=initial_spectrum, parent_state=None, action=None,
        broken_target_count=0, weighted_coverage=0.0, total_syndrome_weight=0,
    )

    beam = [initial_node]; all_nodes = [initial_node]
    visited_states: set[BBState] = {initial.state}

    for depth in range(1, steps + 1):
        print(f"\n[{strategy}] depth {depth}/{steps}")
        for index, node in enumerate(beam):
            print(
                f"  parent[{index}] [[{node.evaluation.n},{node.evaluation.k},"
                f"<={node.evaluation.d_est}]] score={node.spectrum.score:.6g} "
                f"targets={len(node.spectrum.targets)}"
            )

        proposal_by_state: dict[BBState, tuple[SearchNode, StructuralProposal]] = {}
        for parent in beam:
            proposals = enumerate_structural_proposals(
                parent.evaluation, parent.spectrum.targets,
                strategy=strategy, target_k=target_k,
                require_connected=require_connected, include_pairs=include_pairs,
                max_pair_proposals=max_pair_proposals,
                children_per_parent=children_per_parent,
                coverage_gamma=score_gamma, rng=rng,
                visited_states=visited_states,
            )
            print(f"    expanded parent {format_state(parent.evaluation.state)}: {len(proposals)} proposals")
            for proposal in proposals:
                old = proposal_by_state.get(proposal.state)
                if old is None:
                    proposal_by_state[proposal.state] = (parent, proposal); continue
                _, op = old
                if (proposal.weighted_coverage, proposal.broken_target_count,
                    proposal.total_syndrome_weight, -proposal.action.size) > (
                    op.weighted_coverage, op.broken_target_count,
                    op.total_syndrome_weight, -op.action.size):
                    proposal_by_state[proposal.state] = (parent, proposal)

        if not proposal_by_state:
            print(f"[{strategy}] no new proposals; stopping."); break

        candidate_nodes: list[SearchNode] = []
        for candidate_index, (state, (parent, proposal)) in enumerate(proposal_by_state.items(), 1):
            try:
                evaluation, spectrum = get_evaluation(state)
            except (RuntimeError, ValueError) as exc:
                print(f"  skip {proposal.action.label()}: {exc}")
                visited_states.add(state); continue
            if evaluation.d_est < parent.evaluation.d_est - max_distance_drop:
                visited_states.add(state); continue
            node = SearchNode(
                strategy=strategy, depth=depth, evaluation=evaluation,
                spectrum=spectrum, parent_state=parent.evaluation.state,
                action=proposal.action,
                broken_target_count=proposal.broken_target_count,
                weighted_coverage=proposal.weighted_coverage,
                total_syndrome_weight=proposal.total_syndrome_weight,
            )
            candidate_nodes.append(node); visited_states.add(state)
            print(
                f"  child {candidate_index}/{len(proposal_by_state)}: {proposal.action.label()} "
                f"-> [[{evaluation.n},{evaluation.k},<={evaluation.d_est}]] "
                f"score={spectrum.score:.6g} coverage={proposal.weighted_coverage:.3g} "
                f"broken={proposal.broken_target_count}"
            )

        if not candidate_nodes:
            print(f"[{strategy}] no candidates survived search evaluation."); break
        use_coverage = strategy == "targeted"
        candidate_nodes.sort(key=lambda node: node_rank_key(node, use_coverage=use_coverage))
        beam = candidate_nodes[:beam_width]; all_nodes.extend(candidate_nodes)
        print(f"[{strategy}] selected beam:")
        for index, node in enumerate(beam):
            print(
                f"  beam[{index}] [[{node.evaluation.n},{node.evaluation.k},"
                f"<={node.evaluation.d_est}]] score={node.spectrum.score:.6g} "
                f"action={node.action.label() if node.action else '--'}"
            )

    all_nodes.sort(key=lambda node: node_rank_key(node, use_coverage=strategy == "targeted"))
    return all_nodes[0], all_nodes


# =============================================================================
# Initial-code selection and optional exact landscape
# =============================================================================


@dataclass
class LandscapeRecord:
    state: BBState
    k: int
    connected: bool
    evaluation: BBEvaluation | None


@dataclass
class LandscapeSummary:
    total_states: int
    k_histogram: dict[int, int]
    target_k_states: int
    connected_target_k_states: int
    connected_distance_histogram: dict[int, int]
    connected_max_distance: int | None
    records: list[LandscapeRecord]


def exhaustive_landscape(
    *, ell: int, m: int, target_k: int,
    mode: Literal["metadata", "distance"], distance_time_limit: float,
    output_csv: Path | None, evaluation_cache: dict[BBState, BBEvaluation],
) -> LandscapeSummary:
    terms = allowed_pure_axis_terms(ell, m)
    triples = sorted(set(tuple(sorted(t)) for t in combinations(terms, 3)))
    polynomial_cache = {t: polynomial_matrix(ell, m, t) for t in triples}
    k_histogram: defaultdict[int, int] = defaultdict(int)
    records: list[LandscapeRecord] = []; total = 0
    for a_terms in triples:
        a = polynomial_cache[a_terms]
        for b_terms in triples:
            total += 1; b = polynomial_cache[b_terms]
            hx = np.hstack((a, b)).astype(np.uint8)
            hz = np.hstack((b.T, a.T)).astype(np.uint8)
            k, _, _ = compute_k(hx, hz); k_histogram[k] += 1
            if k == target_k:
                state = BBState(ell, m, a_terms, b_terms).canonical()
                records.append(LandscapeRecord(state, k, css_tanner_is_connected(hx, hz), None))
    distance_histogram: defaultdict[int, int] = defaultdict(int)
    if mode == "distance":
        if ell * m > 36:
            raise ValueError("Exact landscape distance mode is intended only for very small lifts (ell*m <= 36).")
        connected_records = [r for r in records if r.connected]
        for index, record in enumerate(connected_records, 1):
            ev = evaluation_cache.get(record.state)
            if ev is None:
                ev = evaluate_bb_state(record.state, distance_time_limit=distance_time_limit)
                evaluation_cache[record.state] = ev
            record.evaluation = ev; distance_histogram[ev.d] += 1
            print(f"  landscape distance {index}/{len(connected_records)}: {format_state(record.state)} -> d{'=' if ev.d_certified else '<='}{ev.d}")
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["A","B","k","connected","d","d_certified","d_x","d_z"])
            writer.writeheader()
            for r in records:
                ev = r.evaluation
                writer.writerow({"A":format_polynomial(r.state.a_terms),"B":format_polynomial(r.state.b_terms),
                    "k":r.k,"connected":r.connected,"d":"" if ev is None else ev.d,
                    "d_certified":"" if ev is None else ev.d_certified,
                    "d_x":"" if ev is None else ev.d_x,"d_z":"" if ev is None else ev.d_z})
    certified_distances = [r.evaluation.d for r in records if r.connected and r.evaluation is not None and r.evaluation.d_certified]
    return LandscapeSummary(total, dict(sorted(k_histogram.items())), len(records),
        sum(r.connected for r in records), dict(sorted(distance_histogram.items())),
        max(certified_distances) if certified_distances else None, records)


def random_bb_state(ell: int, m: int, rng: np.random.Generator) -> BBState:
    terms = allowed_pure_axis_terms(ell, m)
    ai = rng.choice(len(terms), size=3, replace=False)
    bi = rng.choice(len(terms), size=3, replace=False)
    return BBState(ell, m, tuple(terms[int(i)] for i in ai), tuple(terms[int(i)] for i in bi)).canonical()


def find_random_initial_code(
    *, ell: int, m: int, target_k: int, rng: np.random.Generator,
    max_distance: int | None, max_tries: int, require_connected: bool,
    probe_coefficients: np.ndarray, score_gamma: float,
    score_max_weight_offset: int, max_targets_per_type: int,
    decoder_error_rate: float, decoder_max_iter: int, decoder_osd_order: int,
    evaluation_cache: dict[BBState, tuple[SearchEvaluation, SpectrumAnalysis]],
) -> tuple[SearchEvaluation, SpectrumAnalysis]:
    
    tested: set[BBState] = set()

    for attempt in range(1, max_tries + 1):
        state = random_bb_state(ell, m, rng)
        
        if state in tested: continue
        
        tested.add(state)
        _, _, hx, hz = build_bb_checks(state)
        k, _, _ = compute_k(hx, hz)

        if k != target_k: continue
        
        if require_connected and not css_tanner_is_connected(hx, hz): continue

        try:
            pair = evaluate_search_state(
                state, probe_coefficients, gamma=score_gamma,
                max_weight_offset=score_max_weight_offset,
                max_targets_per_type=max_targets_per_type,
                decoder_error_rate=decoder_error_rate,
                decoder_max_iter=decoder_max_iter,
                decoder_osd_order=decoder_osd_order,
            )

        except RuntimeError:
            continue
        
        evaluation_cache[state] = pair
        
        ev, _ = pair
        
        if max_distance is None or ev.d_est <= max_distance:
            print(f"Random initial code found after {attempt} attempts: [[{ev.n},{ev.k},<={ev.d_est}]]")
            return pair
    
    raise RuntimeError(
        f"No {'connected ' if require_connected else ''}random code with k={target_k} "
        f"and sampled distance <= {max_distance} was found in {max_tries} attempts."
    )


# =============================================================================
# Reporting
# =============================================================================


def format_polynomial(terms: tuple[Term, ...]) -> str:
    pieces: list[str] = []
    for u, v in terms:
        if (u, v) == (0, 0): pieces.append("1")
        elif v == 0: pieces.append("x" if u == 1 else f"x^{u}")
        elif u == 0: pieces.append("y" if v == 1 else f"y^{v}")
        else: pieces.append(f"x^{u}y^{v}")
    return " + ".join(pieces)


def format_state(state: BBState) -> str:
    return f"A=({format_polynomial(state.a_terms)}), B=({format_polynomial(state.b_terms)})"


def print_search_evaluation(label: str, evaluation: SearchEvaluation) -> None:
    print(f"\n{label}")
    print(f"  A = {format_polynomial(evaluation.state.a_terms)}")
    print(f"  B = {format_polynomial(evaluation.state.b_terms)}")
    print(f"  Hx shape = {evaluation.hx.shape}, rank = {evaluation.rank_hx}")
    print(f"  Hz shape = {evaluation.hz.shape}, rank = {evaluation.rank_hz}")
    print(f"  connected Tanner graph = {evaluation.connected}")
    print(f"  sampled parameters = [[{evaluation.n},{evaluation.k},<={evaluation.d_est}]], dX<={evaluation.d_x_est}, dZ<={evaluation.d_z_est}")


def print_exact_evaluation(label: str, evaluation: BBEvaluation) -> None:
    print(f"\n{label}")
    print(f"  A = {format_polynomial(evaluation.state.a_terms)}")
    print(f"  B = {format_polynomial(evaluation.state.b_terms)}")
    relation = "=" if evaluation.d_certified else "<="
    print(f"  MILP result: [[{evaluation.n},{evaluation.k},{relation}{evaluation.d}]], dX{'=' if evaluation.d_x_certified else '<='}{evaluation.d_x}, dZ{'=' if evaluation.d_z_certified else '<='}{evaluation.d_z}")


def save_search_csv(nodes: Sequence[SearchNode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fields=["strategy","depth","A","B","n","k","d_est","d_x_est","d_z_est","connected","score","components","target_count","broken_target_count","weighted_coverage","total_syndrome_weight","action","parent_A","parent_B"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for node in nodes:
            ev=node.evaluation; parent=node.parent_state
            writer.writerow({"strategy":node.strategy,"depth":node.depth,
                "A":format_polynomial(ev.state.a_terms),"B":format_polynomial(ev.state.b_terms),
                "n":ev.n,"k":ev.k,"d_est":ev.d_est,"d_x_est":ev.d_x_est,"d_z_est":ev.d_z_est,
                "connected":ev.connected,"score":node.spectrum.score,
                "components":json.dumps(node.spectrum.components,sort_keys=True),
                "target_count":len(node.spectrum.targets),"broken_target_count":node.broken_target_count,
                "weighted_coverage":node.weighted_coverage,"total_syndrome_weight":node.total_syndrome_weight,
                "action":"" if node.action is None else node.action.label(),
                "parent_A":"" if parent is None else format_polynomial(parent.a_terms),
                "parent_B":"" if parent is None else format_polynomial(parent.b_terms)})


# =============================================================================
# CLI
# =============================================================================


def benchmark_state(ell: int, m: int) -> tuple[BBState | None, int | None]:
    if (ell, m) == (6, 6):
        return BBState(6, 6, ((3,0),(0,1),(0,2)), ((0,3),(1,0),(2,0))).canonical(), 6
    if (ell, m) == (12, 12):
        return BBState(12, 12, ((3,0),(0,2),(0,7)), ((0,3),(1,0),(2,0))).canonical(), 18
    return None, None


def main() -> None:
    parser=argparse.ArgumentParser(description="Logical-guided BB search using BP+OSD during search and optional MILP certification.")
    parser.add_argument("--ell",type=int,default=6); parser.add_argument("--m",type=int,default=6)
    parser.add_argument("--seed",type=int,default=7); parser.add_argument("--target-k",type=int,default=12)
    parser.add_argument("--steps",type=int,default=5); parser.add_argument("--beam-width",type=int,default=10)
    parser.add_argument("--children-per-parent",type=int,default=12)
    parser.add_argument("--max-pair-proposals",type=int,default=100)
    parser.add_argument("--disable-pair-actions",action="store_true")
    parser.add_argument("--max-distance-drop",type=int,default=0)
    parser.add_argument("--milp-time-limit",type=float,default=60.0)
    parser.add_argument("--initial-mode",choices=["random","published"],default="random")
    parser.add_argument("--initial-max-distance",type=int,default=np.inf,
        help="Maximum sampled BP+OSD distance upper bound for a random start.")
    parser.add_argument("--random-max-tries",type=int,default=1000)
    parser.add_argument("--allow-disconnected-initial",action="store_true")
    parser.add_argument("--allow-disconnected-candidates",action="store_true")
    parser.add_argument("--score-probes",type=int,default=256,
        help="Positive sample count; negative uses all 2^k-1 logical classes. Zero is not supported in BP+OSD search mode.")
    parser.add_argument("--score-gamma",type=float,default=0.3)
    parser.add_argument("--score-window",type=int,default=2)
    parser.add_argument("--targets-per-type",type=int,default=20)
    parser.add_argument("--decoder-error-rate",type=float,default=0.05)
    parser.add_argument("--decoder-max-iter",type=int,default=1000)
    parser.add_argument("--decoder-osd-order",type=int,default=2)
    parser.add_argument("--run-random-baseline",action="store_true")
    parser.add_argument("--certify-final",action="store_true")
    parser.add_argument("--certify-published",action="store_true")
    parser.add_argument("--landscape-mode",choices=["none","metadata","distance"],default="none")
    parser.add_argument("--output-dir",type=Path,default=Path("optimization/results/bb_logical_guided"))
    args = parser.parse_args()
    if args.score_probes == 0:
        parser.error("--score-probes 0 is incompatible with BP+OSD search evaluation; use a positive value or -1.")

    rng = np.random.default_rng(args.seed); probe_rng=np.random.default_rng(args.seed+1000)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    probe_coefficients = generate_logical_class_probes(args.target_k, args.score_probes, probe_rng)

    exact_cache:dict[BBState,BBEvaluation]={}
    search_cache:dict[BBState,tuple[SearchEvaluation,SpectrumAnalysis]]={}

    landscape_summary = None

    if args.landscape_mode != "none":
        print(f"\nScanning exhaustive ell={args.ell}, m={args.m} landscape...")
        landscape_summary=exhaustive_landscape(ell=args.ell,m=args.m,target_k=args.target_k,
            mode=args.landscape_mode,distance_time_limit=args.milp_time_limit,
            output_csv=args.output_dir/"landscape_target_k.csv",evaluation_cache=exact_cache)
        print(f"  total raw states: {landscape_summary.total_states}")
        print(f"  target-k states: {landscape_summary.target_k_states}")
        print(f"  connected target-k states: {landscape_summary.connected_target_k_states}")

    published, known_d = benchmark_state(args.ell,args.m)
    published_search=None; published_spectrum = None; published_exact = None

    if published is not None:
        _,_,phx,phz = build_bb_checks(published); pk,prx,prz=compute_k(phx,phz)
        print("\nPublished benchmark")
        print(f"  A = {format_polynomial(published.a_terms)}")
        print(f"  B = {format_polynomial(published.b_terms)}")
        print(f"  Hx shape = {phx.shape}, rank = {prx}")
        print(f"  Hz shape = {phz.shape}, rank = {prz}")
        print(f"  known published parameters = [[{phx.shape[1]},{pk},{known_d}]]")

        published_search, published_spectrum = evaluate_search_state(published,probe_coefficients,
            gamma=args.score_gamma, max_weight_offset=args.score_window,
            max_targets_per_type=args.targets_per_type,
            decoder_error_rate=args.decoder_error_rate,decoder_max_iter=args.decoder_max_iter,
            decoder_osd_order=args.decoder_osd_order
        )
        search_cache[published]=(published_search,published_spectrum)
        print(f"  BP+OSD sampled distance upper bound = {published_search.d_est}")
        
        if args.certify_published:
            published_exact = evaluate_bb_state(published,distance_time_limit=args.milp_time_limit)
            print_exact_evaluation("Published benchmark MILP",published_exact)
    
    elif args.initial_mode == "published":
        parser.error("No built-in published benchmark for this ell,m pair.")

    if args.initial_mode == "published":
        assert published_search is not None and published_spectrum is not None
        initial,initial_spectrum=published_search,published_spectrum
    else:
        initial, initial_spectrum = find_random_initial_code(
            ell=args.ell, m=args.m, target_k=args.target_k, rng=rng,
            max_distance=args.initial_max_distance, max_tries=args.random_max_tries,
            require_connected=not args.allow_disconnected_initial,
            probe_coefficients=probe_coefficients, score_gamma=args.score_gamma,
            score_max_weight_offset=args.score_window, max_targets_per_type=args.targets_per_type,
            decoder_error_rate=args.decoder_error_rate, decoder_max_iter=args.decoder_max_iter,
            decoder_osd_order=args.decoder_osd_order, evaluation_cache=search_cache)
    
    print_search_evaluation("Initial search code",initial)

    common = dict(steps=args.steps, beam_width=args.beam_width,
        children_per_parent=args.children_per_parent,include_pairs=not args.disable_pair_actions,
        max_pair_proposals=args.max_pair_proposals,target_k=args.target_k,
        require_connected=not args.allow_disconnected_candidates,
        max_distance_drop=args.max_distance_drop,probe_coefficients=probe_coefficients,
        score_gamma=args.score_gamma,score_max_weight_offset=args.score_window,
        max_targets_per_type=args.targets_per_type,decoder_error_rate=args.decoder_error_rate,
        decoder_max_iter=args.decoder_max_iter,decoder_osd_order=args.decoder_osd_order,
        evaluation_cache=search_cache
    )

    targeted_best, targeted_nodes = run_beam_search(initial,initial_spectrum,strategy="targeted",
        rng=np.random.default_rng(args.seed+2000),**common)
    print_search_evaluation("Best targeted-search code",targeted_best.evaluation)
    print(f"  sampled relative score = {targeted_best.spectrum.score:.6g}, components={targeted_best.spectrum.components}")
    save_search_csv(targeted_nodes,args.output_dir/"targeted_search.csv")

    random_best = None; random_nodes=[]

    if args.run_random_baseline:
        random_best, random_nodes = run_beam_search(initial,initial_spectrum,strategy="random",
            rng=np.random.default_rng(args.seed+3000),**common)
        print_search_evaluation("Best random-proposal code",random_best.evaluation)
        print(f"  sampled relative score = {random_best.spectrum.score:.6g}, components={random_best.spectrum.components}")
        save_search_csv(random_nodes,args.output_dir/"random_search.csv")

    certifications={}
    if args.certify_final:
        for label,node in [("targeted",targeted_best),("random",random_best)]:
            if node is None: continue
            try:
                ex=evaluate_bb_state(node.evaluation.state,distance_time_limit=args.milp_time_limit)
                certifications[label]=ex
                print_exact_evaluation(f"Final {label} MILP",ex)
            except RuntimeError as exc:
                print(f"Final {label} MILP produced no incumbent: {exc}")

    def search_json(ev:SearchEvaluation,spec:SpectrumAnalysis)->dict:
        return {"A":format_polynomial(ev.state.a_terms),"B":format_polynomial(ev.state.b_terms),
            "n":ev.n,"k":ev.k,"d_est":ev.d_est,"d_x_est":ev.d_x_est,"d_z_est":ev.d_z_est,
            "connected":ev.connected,"score":spec.score,"components":spec.components}
    summary={"settings":{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},
        "published":None if published is None else {"A":format_polynomial(published.a_terms),"B":format_polynomial(published.b_terms),"known_d":known_d,
            "search":None if published_search is None else search_json(published_search,published_spectrum)},
        "initial":search_json(initial,initial_spectrum),
        "targeted_best":search_json(targeted_best.evaluation,targeted_best.spectrum),
        "random_best":None if random_best is None else search_json(random_best.evaluation,random_best.spectrum),
        "certifications":{label:{"d":ev.d,"d_certified":ev.d_certified,"d_x":ev.d_x,"d_x_certified":ev.d_x_certified,"d_z":ev.d_z,"d_z_certified":ev.d_z_certified} for label,ev in certifications.items()}}
    with (args.output_dir/"summary.json").open("w") as handle: json.dump(summary,handle,indent=2,sort_keys=True)
    print(f"\nSaved results to {args.output_dir}")
    print(f"Comparison (BP+OSD upper bounds): initial<={initial.d_est}, targeted<={targeted_best.evaluation.d_est}" + ("" if random_best is None else f", random<={random_best.evaluation.d_est}") + ("" if known_d is None else f", published known d={known_d}"))


if __name__ == "__main__":
    main()
