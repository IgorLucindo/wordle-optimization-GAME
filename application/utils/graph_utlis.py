import networkx as nx
import matplotlib.pyplot as plt


def show_tree_graph(G, root, plot_flag=True):
    """
    Plot tree graph with faint edges and no labels
    """
    if not plot_flag:
        return

    T = nx.dfs_tree(G, source=root)
    
    levels = {vertex: nx.shortest_path_length(T, source=root, target=vertex) for vertex in T.nodes}
    nx.set_node_attributes(T, values=levels, name='level')
    pos = nx.multipartite_layout(T, subset_key='level')

    # Set title
    ax = plt.gca()
    
    # Draw graph without any labels
    nx.draw(
        T,
        pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        alpha=0.8,
        width=0.5,
        ax=ax
    )

    plt.title(f"Guess Tree (#vertices: {len(T.nodes)}, height: {max(levels.values())})")
    plt.show()