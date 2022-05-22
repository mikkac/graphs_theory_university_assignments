import argparse

import sys, os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from list_03.list_03 import WeightedGraph


def setup_args():
    """Setup argparse to take graph name argument."""
    parser = argparse.ArgumentParser(description="Find an Eularian Cicruit.")
    parser.add_argument("graph_path", type=str, nargs="?", help="Path of graph file")
    parser.add_argument("start_node", type=int, nargs="?", help="Start node")
    parser.add_argument("end_node", type=int, nargs="?", help="End node")
    args = parser.parse_args()
    return args.graph_path, args.start_node, args.end_node


def dijkstra(graph_file: str, start_node: int, end_node: int) -> None:
    """Dijkstra algorithm"""
    graph: WeightedGraph = WeightedGraph(graph_file)

    cost: int = graph.get_shortest_path(start_node, end_node)
    print(
        f"Najkrotsza droga miedzy wierzcholkami {start_node} i {end_node} wynosi: {cost}"
    )


def main():
    graph_path, start_node, end_node = setup_args()
    dijkstra(graph_path, start_node, end_node)


main()
