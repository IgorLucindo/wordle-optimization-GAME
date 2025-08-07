from utils.graph_utlis import *
import json
import ast


def main():
    # Load graph
    with open('dataset/decision_tree.json', 'r') as f:
        tree = json.load(f)
        tree['nodes'] = {ast.literal_eval(k): v for k, v in tree['nodes'].items()}

    # Create graph
    G = nx.Graph()
    G.add_edges_from(tree['edges'])

    # Show graph
    show_tree_graph(G, tree['root'])


if __name__ == "__main__":
    main()