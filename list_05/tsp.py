import argparse

import sys, os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from list_03.list_03 import WeightedGraph


def setup_args():
    """Setup argparse to take graph name argument."""
    parser = argparse.ArgumentParser(description="Find an Eularian Cicruit.")
    parser.add_argument("graph_path", type=str, nargs="?", help="Path of graph file")
    parser.add_argument("start_node", type=int, nargs="?", help="Start node")
    args = parser.parse_args()
    return args.graph_path, args.start_node


def tsp(graph_file: str, start_node: int) -> None:
    """Traveling Salesman Problem"""
    graph: WeightedGraph = WeightedGraph(graph_file)

    cost, best_path = graph.naive_tsp(start_node)
    print(
        f"Problem komiwojazera - najkrotsza droga dla wierzcholka {start_node} wynosi: {cost}"
    )


def main():
    graph_path, start_node = setup_args()
    tsp(graph_path, start_node)


main()
