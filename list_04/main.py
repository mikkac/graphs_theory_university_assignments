#!/usr/bin/env python
"""
Solve the Chinese-Postman problem.

For a given graph, determine the minimum amount of backtracking
required to complete an Eularian circuit.

"""
import os
import pathlib
import eularian, network


def chinese_postman_problem():
    graph_path: str = os.path.join(pathlib.Path(__file__).parent.resolve(), "g1.txt")
    start_node: int = 4
    print(f"Loading graph: {graph_path}")

    original_graph: network.Graph = network.Graph(graph_path)

    print(f"<{len(original_graph)}> edges")
    if not original_graph.is_eularian:
        print("Converting to Eularian path...")
        graph, num_dead_ends = eularian.make_eularian(original_graph)
        print("Conversion complete")
        print(f"\tAdded {len(graph) - len(original_graph) + num_dead_ends} edges")
        print(f"\tTotal cost is {graph.total_cost}")
    else:
        graph = original_graph

    print("Attempting to solve Eularian Circuit...")
    route, cost = eularian.eularian_path(graph, start=start_node)
    if not route:
        print("\tGave up")
    else:
        print(f"Solution (cost={cost}):\n\t{route}")

    original_graph.dump_graph(
        f"{graph_path.split('.')[0]}_task_01.jpg",
        f"{graph_path.split('.')[0]}_task_01_.dot",
    )


if __name__ == "__main__":
    chinese_postman_problem()
