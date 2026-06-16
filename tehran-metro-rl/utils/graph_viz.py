import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(graph, path=None):
    G = nx.Graph()

    for node, neighbors in graph.items():
        for n in neighbors:
            G.add_edge(node, n)

    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=1200,
        font_size=8,
        edge_color="gray"
    )

    if path:
        edges = list(zip(path, path[1:]))

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=path,
            node_color="orange",
            node_size=1400
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=3,
            edge_color="red"
        )

    plt.title("Metro Graph + Learned Path (Q-learning)")
    plt.show()