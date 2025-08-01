from utils.graph_utlis import *
import json


def main():
    # Set path
    graph_path = 'application/results/graphs/'

    # Load graph
    with open(graph_path + 'guess_tree.json', 'r') as f:
        tree = json.load(f)

    # Create graph
    G = nx.Graph()
    G.add_edges_from(tree['edges'])

    # Show graph
    show_tree_graph(G, tree['root'])


if __name__ == "__main__":
    main()