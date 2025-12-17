from pyvis.network import Network
import networkx as nx


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
        data=True) if d.get('bipartite') == 1]  # Checks
    bottom_nodes = [n for n, d in G.nodes(
        data=True) if d.get('bipartite') == 0]  # Symbols

    top_nodes.sort()
    bottom_nodes.sort()

    # 3. Add Nodes (Explicitly casting to int)
    canvas_width = 1000

    # --- Add Symbol Nodes (Top visual row: y=100) ---
    spacing_x_symbols = canvas_width / (len(bottom_nodes) + 1)
    for i, node in enumerate(bottom_nodes):
        x_pos = (i + 1) * spacing_x_symbols
        y_pos = 100

        net.add_node(
            int(node),                 # <--- FIX: Cast numpy.int64 to int
            label=f"S{i}",
            title=f"Symbol Node {i}",
            color="#87CEEB",           # SkyBlue
            x=x_pos,
            y=y_pos,
            physics=False,
            shape="circle"
        )

    # --- Add Check Nodes (Bottom visual row: y=400) ---
    spacing_x_checks = canvas_width / (len(top_nodes) + 1)
    for i, node in enumerate(top_nodes):
        x_pos = (i + 1) * spacing_x_checks
        y_pos = 400

        net.add_node(
            int(node),                 # <--- FIX: Cast numpy.int64 to int
            label=f"C{i}",
            title=f"Check Node {i}",
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


# --- Run it ---
# draw_interactive_tanner_graph(best_state)
