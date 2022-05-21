#!/usr/bin/env python
"""
Solve the Chinese-Postman problem.

For a given graph, determine the minimum amount of backtracking
required to complete an Eularian circuit.

"""
import eularian, network


def main():
    """Make it so."""
    graph_path: str = "g1.txt"
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
    route, attempts = eularian.eularian_path(graph, start=start_node)
    if not route:
        print(f"\tGave up after <{attempts}> attempts.")
    else:
        print(f"\tSolved in <{attempts}> attempts")
        print(f"Solution:\n\t{route}")

    original_graph.dump_graph(
        f"task_01_{graph_path.split('.')[0]}.jpg",
        f"task_01_{graph_path.split('.')[0]}.dot",
    )


if __name__ == "__main__":
    main()
