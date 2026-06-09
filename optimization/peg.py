"""
Python translation of Anil Uzumcuoglu's PEG / TannerGraph JavaScript code.

By default this follows the deterministic tie-breaking behavior of the JS code:
when several check nodes have the same lowest degree, it chooses the first one.
Set random_ties=True to generate different PEG matrices and then filter for
full row rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import argparse
import random
import numpy as np


@dataclass(eq=False)
class Node:
    matrix_idx: int
    id: int
    label: str
    group: str  # "symbol" or "check"
    connections: list["Node"] = field(default_factory=list)

    @property
    def degree(self) -> int:
        return len(self.connections)


@dataclass
class TreeNode:
    ref: Node
    id: int
    label: str
    group: str
    level: int
    children: list["TreeNode"] = field(default_factory=list)


class TannerGraph:
    def __init__(self, matrix: np.ndarray | list[list[int]]):
        self.matrix = np.asarray(matrix, dtype=np.uint8).copy()
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be 2-dimensional")

        m, n = self.matrix.shape

        self.symbol_nodes: list[Node] = [
            Node(matrix_idx=j, id=j, label=f"S{j}", group="symbol")
            for j in range(n)
        ]
        self.check_nodes: list[Node] = [
            Node(matrix_idx=i, id=i + n, label=f"C{i}", group="check")
            for i in range(m)
        ]

        self._nodes_by_id: dict[int, Node] = {
            node.id: node for node in self.symbol_nodes + self.check_nodes
        }
        self.edges: list[dict[str, int]] = []

        for i in range(m):
            for j in range(n):
                if int(self.matrix[i, j]) == 1:
                    check = self.check_nodes[i]
                    symbol = self.symbol_nodes[j]
                    check.connections.append(symbol)
                    symbol.connections.append(check)
                    self.edges.append({"from": check.id, "to": symbol.id})

    def get_node(self, node_id: int) -> Optional[Node]:
        return self._nodes_by_id.get(node_id)

    def get_clone(self) -> "TannerGraph":
        return TannerGraph(self.matrix.copy())

    def create_edge(self, symbol_node_id: int, check_node_id: int) -> None:
        symbol = self.get_node(symbol_node_id)
        check = self.get_node(check_node_id)
        if symbol is None or check is None:
            raise ValueError("invalid symbol or check node id")
        if symbol.group != "symbol" or check.group != "check":
            raise ValueError("create_edge expects (symbol_node_id, check_node_id)")

        # Match the JS behavior. PEG selection should normally avoid duplicates.
        symbol.connections.append(check)
        check.connections.append(symbol)
        self.edges.append({"from": check.id, "to": symbol.id})
        self.matrix[check.matrix_idx, symbol.matrix_idx] = 1

    @staticmethod
    def _choose_lowest_degree(
        nodes: list[Node],
        *,
        random_ties: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Node:
        if not nodes:
            raise ValueError("cannot choose from an empty node list")
        min_degree = min(node.degree for node in nodes)
        candidates = [node for node in nodes if node.degree == min_degree]
        if random_ties:
            if rng is None:
                rng = random.Random()
            return rng.choice(candidates)
        return candidates[0]

    def get_check_node_with_lowest_degree(
        self,
        *,
        random_ties: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Node:
        return self._choose_lowest_degree(
            self.check_nodes, random_ties=random_ties, rng=rng
        )

    def get_symbol_node_with_lowest_degree(
        self,
        *,
        random_ties: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Node:
        return self._choose_lowest_degree(
            self.symbol_nodes, random_ties=random_ties, rng=rng
        )

    def get_subgraph(self, node_id: int, depth: int) -> "SubGraph":
        return SubGraph(self, node_id, depth)


class SubGraph:
    def __init__(self, tanner_graph: TannerGraph, root_node_id: int, depth: int):
        if depth < 0:
            raise ValueError("depth cannot be negative")

        root = tanner_graph.get_node(root_node_id)
        if root is None:
            raise ValueError(f"node id {root_node_id} not found")

        self.root_node = root
        self._tanner_graph = tanner_graph
        self.tree_root = TreeNode(
            ref=root,
            id=root.id,
            label=root.label,
            group=root.group,
            level=0,
        )

        level = 0
        used_ids = {self.tree_root.id}
        queue = [self.tree_root]

        while queue and level < depth:
            level += 1
            level_queue: list[TreeNode] = []

            for node in queue:
                child_nodes: list[TreeNode] = []
                for conn in node.ref.connections:
                    child_nodes.append(
                        TreeNode(
                            ref=conn,
                            id=conn.id,
                            label=conn.label,
                            group=conn.group,
                            level=level,
                        )
                    )

                node.children = [child for child in child_nodes if child.id not in used_ids]
                used_ids.update(child.id for child in node.children)
                level_queue.extend(node.children)

            queue = level_queue

        # This matches the JS behavior: this is the level reached, not necessarily depth.
        self.level = level

    def covered_check_nodes(self) -> list[TreeNode]:
        covered: list[TreeNode] = []
        queue = [self.tree_root]

        while queue:
            node = queue.pop(0)
            if node.group == "check":
                covered.append(node)
            queue.extend(node.children)

        return covered

    def all_check_nodes_covered(self) -> bool:
        return len(self.covered_check_nodes()) == len(self._tanner_graph.check_nodes)

    def get_uc_check_node_with_lowest_degree(
        self,
        *,
        random_ties: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Node:
        covered_ids = {node.id for node in self.covered_check_nodes()}
        uncovered = [
            node for node in self._tanner_graph.check_nodes if node.id not in covered_ids
        ]
        return TannerGraph._choose_lowest_degree(
            uncovered, random_ties=random_ties, rng=rng
        )


class PEG:
    def __init__(
        self,
        *,
        random_ties: bool = False,
        seed: Optional[int] = None,
        hook: Optional[Callable[[TannerGraph], None]] = None,
    ):
        self.random_ties = random_ties
        self.rng = random.Random(seed)
        self.hook = hook

    def create(
        self,
        *,
        check_node_number: int,
        symbol_node_number: int,
        symbol_node_degrees: list[int],
    ) -> TannerGraph:
        if len(symbol_node_degrees) != symbol_node_number:
            raise ValueError(
                "len(symbol_node_degrees) must equal symbol_node_number"
            )

        parity_check_matrix = np.zeros(
            (check_node_number, symbol_node_number), dtype=np.uint8
        )
        tanner_graph = TannerGraph(parity_check_matrix)

        for index, degree in enumerate(symbol_node_degrees):
            symbol_node = tanner_graph.get_node(index)
            if symbol_node is None:
                raise RuntimeError("symbol node not found")

            for edge_index in range(int(degree)):
                if edge_index == 0:
                    lowest = tanner_graph.get_check_node_with_lowest_degree(
                        random_ties=self.random_ties, rng=self.rng
                    )
                    tanner_graph.create_edge(symbol_node.id, lowest.id)
                    if self.hook:
                        self.hook(tanner_graph)
                else:
                    depth = 0
                    current_subgraph = tanner_graph.get_subgraph(symbol_node.id, depth)

                    while True:
                        if current_subgraph.all_check_nodes_covered():
                            previous_subgraph = tanner_graph.get_subgraph(
                                symbol_node.id, depth - 1
                            )
                            lowest = previous_subgraph.get_uc_check_node_with_lowest_degree(
                                random_ties=self.random_ties, rng=self.rng
                            )
                            tanner_graph.create_edge(symbol_node.id, lowest.id)
                            if self.hook:
                                self.hook(tanner_graph)
                            break

                        depth += 1
                        next_subgraph = tanner_graph.get_subgraph(symbol_node.id, depth)

                        if next_subgraph.level == current_subgraph.level:
                            lowest = current_subgraph.get_uc_check_node_with_lowest_degree(
                                random_ties=self.random_ties, rng=self.rng
                            )
                            tanner_graph.create_edge(symbol_node.id, lowest.id)
                            if self.hook:
                                self.hook(tanner_graph)
                            break

                        current_subgraph = next_subgraph

        return tanner_graph


def create_peg_matrix(
    *,
    symbol_node_number: int,
    check_node_number: int,
    symbol_node_degrees: list[int],
    random_ties: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    peg = PEG(random_ties=random_ties, seed=seed)
    graph = peg.create(
        check_node_number=check_node_number,
        symbol_node_number=symbol_node_number,
        symbol_node_degrees=symbol_node_degrees,
    )
    return graph.matrix.copy()


def gf2_rank(H: np.ndarray) -> int:
    """Rank over GF(2), implemented without external LDPC dependencies."""
    A = np.asarray(H, dtype=np.uint8).copy() & 1
    m, n = A.shape
    rank = 0

    for col in range(n):
        pivot_rows = np.where(A[rank:, col] == 1)[0]
        if len(pivot_rows) == 0:
            continue
        pivot = rank + int(pivot_rows[0])

        if pivot != rank:
            A[[rank, pivot]] = A[[pivot, rank]]

        for row in range(m):
            if row != rank and A[row, col]:
                A[row, :] ^= A[rank, :]

        rank += 1
        if rank == m:
            break

    return int(rank)


def is_full_row_rank(H: np.ndarray) -> bool:
    H = np.asarray(H, dtype=np.uint8)
    return gf2_rank(H) == H.shape[0]


def matrix_to_support_lines(H: np.ndarray) -> list[str]:
    H = np.asarray(H, dtype=np.uint8)
    m, n = H.shape
    lines = [f"{m} {n}"]
    for i in range(m):
        cols = np.where(H[i] == 1)[0]
        lines.append(" ".join(str(int(c)) for c in cols))
    return lines


def write_support_file(H: np.ndarray, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(matrix_to_support_lines(H)))
        f.write("\n")


def print_binary_matrix(H: np.ndarray) -> None:
    H = np.asarray(H, dtype=np.uint8)
    for row in H:
        print(" ".join(str(int(x)) for x in row))


def generate_full_row_rank_peg(
    *,
    symbol_node_number: int,
    check_node_number: int,
    symbol_node_degrees: list[int],
    max_tries: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """
    Generate randomized-tie PEG matrices until one has full row rank.

    Returns
    -------
    H, accepted_seed
    """
    for t in range(max_tries):
        trial_seed = seed + t
        H = create_peg_matrix(
            symbol_node_number=symbol_node_number,
            check_node_number=check_node_number,
            symbol_node_degrees=symbol_node_degrees,
            random_ties=True,
            seed=trial_seed,
        )
        if is_full_row_rank(H):
            return H, trial_seed

    raise RuntimeError(
        f"No full-row-rank matrix found in {max_tries} tries. "
        "Increase max_tries or change the degree sequence."
    )


def _parse_degrees(text: str, n: int) -> list[int]:
    text = text.strip()
    if "," in text:
        degrees = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        # A single integer means a regular degree sequence.
        degrees = [int(text)] * n

    if len(degrees) != n:
        raise ValueError(f"expected {n} degrees, got {len(degrees)}")
    return degrees


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive Edge Growth PEG generator")
    parser.add_argument("--n", type=int, default=28, help="number of symbol nodes")
    parser.add_argument("--m", type=int, default=21, help="number of check nodes")
    parser.add_argument(
        "--degrees",
        type=str,
        default="3",
        help="single degree, e.g. '3', or comma-separated degree sequence",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-ties", action="store_true")
    parser.add_argument("--full-rank", action="store_true")
    parser.add_argument("--max-tries", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--print-binary", action="store_true")

    args = parser.parse_args()
    degrees = _parse_degrees(args.degrees, args.n)

    if args.full_rank:
        H, used_seed = generate_full_row_rank_peg(
            symbol_node_number=args.n,
            check_node_number=args.m,
            symbol_node_degrees=degrees,
            max_tries=args.max_tries,
            seed=args.seed,
        )
        print(f"accepted_seed={used_seed}")
    else:
        H = create_peg_matrix(
            symbol_node_number=args.n,
            check_node_number=args.m,
            symbol_node_degrees=degrees,
            random_ties=args.random_ties,
            seed=args.seed,
        )

    rank = gf2_rank(H)
    print(f"shape={H.shape[0]}x{H.shape[1]}")
    print(f"rank={rank}")
    print(f"full_row_rank={rank == H.shape[0]}")
    print(f"k={H.shape[1] - rank}")
    print(f"k_T={H.shape[0] - rank}")

    if args.output:
        write_support_file(H, args.output)
        print(f"wrote {args.output}")

    if args.print_binary:
        print_binary_matrix(H)
    else:
        print("\n".join(matrix_to_support_lines(H)))


if __name__ == "__main__":
    main()
