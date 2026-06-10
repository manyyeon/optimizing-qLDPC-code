from pyvis.network import Network
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

def node_v(j: int) -> str:
    return rf"$v_{j}$"


def node_c(i: int) -> str:
    return rf"$c_{i}$"

def draw_interactive_tanner_graph(G, filename="tanner_graph.html"):
    """
    Generates an interactive HTML Tanner graph using PyVis.
    Fixes the numpy.int64 vs int AssertionError.
    """

    # 1. Initialize PyVis Network
    net = Network(height="600px", width="100%",
                  bgcolor="white", font_color="black")

    # 2. Separate Nodes
    top_nodes = [n for n, d in G.nodes(
        data=True) if d.get('bipartite') == 1]  # Variable nodes
    bottom_nodes = [n for n, d in G.nodes(
        data=True) if d.get('bipartite') == 0]  # Check nodes

    top_nodes.sort()
    bottom_nodes.sort()

    # 3. Add Nodes (Explicitly casting to int)
    canvas_width = 10000

    # --- Add Check Nodes (Top visual row: y=100) ---
    spacing_x_checks = canvas_width / (len(bottom_nodes) + 1)
    for i, node in enumerate(bottom_nodes):
        x_pos = (i + 1) * spacing_x_checks
        y_pos = 100

        net.add_node(
            int(node),                 # <--- FIX: Cast numpy.int64 to int
            label=f"C{i}",
            title=f"Check Node {i}",
            color="#87CEEB",           # SkyBlue
            x=x_pos,
            y=y_pos,
            physics=False,
            shape="circle"
        )

    # --- Add Variable Nodes (Bottom visual row: y=400) ---
    spacing_x_variables = canvas_width / (len(top_nodes) + 1)
    for i, node in enumerate(top_nodes):
        x_pos = (i + 1) * spacing_x_variables
        y_pos = 400

        net.add_node(
            int(node),                 # <--- FIX: Cast numpy.int64 to int
            label=f"V{i}",
            title=f"Variable Node {i}",
            color="#F08080",           # LightCoral
            x=x_pos,
            y=y_pos,
            physics=False,
            shape="box"
        )

    # 4. Add Edges (Must also cast to int to match the nodes)
    for u, v in G.edges():
        net.add_edge(int(u), int(v), color="lightgrey")

    # 5. Generate Options
    net.set_options("""
    var options = {
      "interaction": {
        "dragNodes": true,
        "zoomView": true,
        "hover": true
      },
      "physics": {
        "enabled": false
      }
    }
    """)

    # 6. Save and Show
    net.save_graph(filename)
    print(f"Graph saved to {filename}. Open this file in your browser.")

#!/usr/bin/env python3
"""
Draw the Tanner graph of the classical [7,4,3] Hamming code.

A Tanner graph is a bipartite graph:
  - variable nodes v1,...,v7 represent codeword bits
  - check nodes c1,c2,c3 represent parity-check equations
  - an edge c_i -- v_j exists when H[i,j] = 1

This script saves the figure as PNG and SVG.
"""

def hamming_743_parity_check_matrix() -> np.ndarray:
    """
    Standard parity-check matrix for the [7,4,3] Hamming code.

    The columns are the nonzero 3-bit binary vectors:
        001, 010, 011, 100, 101, 110, 111
    up to row ordering. Any column permutation gives an equivalent Hamming code.
    """
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=int,
    )

def build_tanner_graph(H: np.ndarray) -> nx.Graph:
    m, n = H.shape
    G = nx.Graph()

    for j in range(n):
        G.add_node(node_v(j + 1), bipartite="variable")
    for i in range(m):
        G.add_node(node_c(i + 1), bipartite="check")

    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                G.add_edge(node_c(i + 1), node_v(j + 1))

    return G


def fixed_bipartite_layout(H: np.ndarray) -> dict[str, tuple[float, float]]:
    m, n = H.shape
    pos = {}
    for j in range(n):
        pos[node_v(j + 1)] = (j, 0.0)

    check_x = np.linspace(1.0, n - 2.0, m)
    for i, x in enumerate(check_x):
        pos[node_c(i + 1)] = (float(x), 1.8)

    return pos

def save_all(fig, output_base: str | Path, dpi: int = 300) -> None:
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for ext in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(ext), dpi=dpi, bbox_inches="tight")
        print(f"Saved {output_base.with_suffix(ext)}")
    plt.close(fig)

def draw_nodes_and_labels(G, pos, H, ax, error_support=None) -> None:
    error_support = error_support or set()

    variable_nodes = [node_v(j + 1) for j in range(H.shape[1])]
    check_nodes = [node_c(i + 1) for i in range(H.shape[0])]
    normal_variable_nodes = [v for v in variable_nodes if v not in error_support]
    error_variable_nodes = [v for v in variable_nodes if v in error_support]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=normal_variable_nodes,
        node_shape="o",
        node_size=780,
        node_color="#F08080",
        edgecolors="none",
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=error_variable_nodes,
        node_shape="o",
        node_size=850,
        node_color="#F08080",
        edgecolors="#B00000",
        linewidths=2.3,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=check_nodes,
        node_shape="s",
        node_size=780,
        node_color="#87CEEB",
        edgecolors="none",
        ax=ax,
    )

    labels = {node: node for node in G.nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=15,
        font_weight="bold",
        ax=ax,
    )

    for v in sorted(error_support):
        x, y = pos[v]
        ax.text(
            x,
            y - 0.45,
            r"$E$",
            color="#B00000",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
        )


def draw_plain_tanner_graph(output_base: str | Path, dpi: int = 300) -> None:
    H = hamming_743_parity_check_matrix()
    G = build_tanner_graph(H)
    pos = fixed_bipartite_layout(H)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=1.7,
        alpha=0.7,
        edge_color="0.55",
    )
    draw_nodes_and_labels(G, pos, H, ax)

    ax.set_axis_off()
    ax.set_xlim(-0.6, H.shape[1] - 0.4)
    ax.set_ylim(-0.55, 2.25)
    fig.tight_layout(pad=0.1)
    save_all(fig, output_base, dpi=dpi)


def draw_edge_swap_action(output_base: str | Path, dpi: int = 300) -> None:
    H = hamming_743_parity_check_matrix()
    G = build_tanner_graph(H)
    pos = fixed_bipartite_layout(H)

    # Weight-3 undetectable pattern:
    # H[:,v3] + H[:,v5] + H[:,v6] = 0 over F_2.
    error_support = {node_v(3), node_v(5), node_v(6)}

    # Example edge swap:
    # remove (v3,c1), (v4,c3); add (v3,c3), (v4,c1)
    removed_edges = [(node_c(1), node_v(3)), (node_c(3), node_v(4))]
    added_edges = [(node_c(3), node_v(3)), (node_c(1), node_v(4))]

    removed_edge_set = {frozenset(e) for e in removed_edges}
    gray_edges = [e for e in G.edges() if frozenset(e) not in removed_edge_set]

    fig, ax = plt.subplots(figsize=(7.2, 3.25))

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=gray_edges,
        ax=ax,
        width=1.4,
        alpha=0.38,
        edge_color="0.55",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=removed_edges,
        ax=ax,
        width=2.4,
        alpha=0.95,
        edge_color="#D00000",
        style="dashed",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=added_edges,
        ax=ax,
        width=2.4,
        alpha=0.95,
        edge_color="#007A5E",
        style="solid",
    )

    draw_nodes_and_labels(G, pos, H, ax, error_support=error_support)

    legend_handles = [
        Line2D([0], [0], color="#D00000", lw=2.4, linestyle="--", label="removed edge"),
        Line2D([0], [0], color="#007A5E", lw=2.4, linestyle="-", label="added edge"),
        Line2D([0], [0], color="0.55", lw=1.4, alpha=0.5, label="unchanged edge"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.5,
    )

    ax.set_axis_off()
    ax.set_xlim(-0.6, H.shape[1] - 0.4)
    ax.set_ylim(-0.75, 2.25)
    fig.tight_layout(pad=0.1)
    save_all(fig, output_base, dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="figures/tanner_graph", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--only", choices=["plain", "action", "both"], default="both")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.only in {"plain", "both"}:
        draw_plain_tanner_graph(output_dir / "hamming_743_tanner_graph", dpi=args.dpi)
    if args.only in {"action", "both"}:
        draw_edge_swap_action(output_dir / "hamming_743_edge_swap_action", dpi=args.dpi)


if __name__ == "__main__":
    main()
