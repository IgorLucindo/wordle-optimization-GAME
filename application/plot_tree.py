from utils.instance_utils import *
from utils.graph_utlis import *
import networkx as nx


def main():
    # Load graph
    tree = get_guess_tree()

    # Create graph
    G = nx.Graph()
    G.add_edges_from(tree['edges'])

    # Show graph
    show_tree_graph(G, tree['root'])


if __name__ == "__main__":
    main()