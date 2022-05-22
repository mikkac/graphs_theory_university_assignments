import argparse

import sys, os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# from list_04 import eularian, network
from list_04 import eularian, network

# import listnetwork


def setup_args():
    """Setup argparse to take graph name argument."""
    parser = argparse.ArgumentParser(description="Find an Eularian Cicruit.")
    parser.add_argument("graph_path", type=str, nargs="?", help="Path of graph file")
    parser.add_argument("start_node", type=int, nargs="?", help="Start node")
    args = parser.parse_args()
    return args.graph_path, args.start_node


def cpp(graph_file: str, start_node: int) -> None:
    """Chinese Postman Problem"""
    original_graph: network.Graph = network.Graph(graph_file)

    if not original_graph.is_eularian:
        graph, num_dead_ends = eularian.make_eularian(original_graph)
    else:
        graph = original_graph

    route, cost = eularian.eularian_path(graph, start=start_node)
    if not route:
        print("Nie udało się wyznaczyć optymalnego rozwiązania")
    else:
        print(
            f"Problem chinskiego listonosza - najkrotsza trasa dla wierzcholka {start_node} wynosi: {cost}"
        )


def main():
    graph_path, start_node = setup_args()
    cpp(graph_path, start_node)


main()
