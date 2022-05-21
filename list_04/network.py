from __future__ import annotations
import copy
from typing import List, Dict, Tuple, Set

from networkx.drawing.nx_agraph import to_agraph, write_dot
from networkx import MultiGraph


class Graph:
    """Abstract representation of a graph."""

    def __init__(self, graph_file_path):
        self.edges: dict = {}
        graph_as_str: str = self._read_graph_file(graph_file_path)

        self._load_edges(graph_as_str)

    @staticmethod
    def _read_graph_file(graph_file_path: str) -> str:
        """
        Reads graph from file associated with `graph_file_path`
        Graph file is expected to be in form of list of edges (lines 1-n):
        Line 0: <number of vertices> <number of edges>
        Line 1-n: <vertex 1> <vertex 2>
        ...
        """
        with open(graph_file_path, mode="r", encoding="utf-8") as graph_file:
            lines = graph_file.readlines()
        return lines

    def _load_edges(self, graph_as_str: str) -> None:
        """
        Fetches list of weighted edges from read file
        """
        for line in graph_as_str[1:]:
            self.add_edge(*[int(x) for x in line.split()])

    def __repr__(self) -> str:
        return "Graph({})".format(str(self.edges))

    def add_edge(self, head: int, tail: int, weight: int) -> None:
        """Adds an Edge to our graph."""
        self.edges[len(self.edges)] = Edge(head, tail, weight)

    def add_edges(self, edges: List[Tuple[int, int, int]]):
        """Add a list of edges."""
        for head, tail, weight in edges:
            self.add_edge(head, tail, weight)  # edge is a tuple of data

    def remove_edges(self, nodes: List[int]):
        """Removes a list of edges associated with `nodes`."""
        for node in nodes:
            self.remove_edge(node)

    def remove_edge(self, node):
        """Remove an edge associated with edge, plus node if it's disconnected."""
        del self.edges[node]  # Remove by key

    @property
    def nodes(self):
        """Return a set of all node indices in this graph."""
        return set(
            [node for edge in self.edges.values() for node in (edge.head, edge.tail)]
        )

    @property
    def node_keys(self) -> List[int]:
        """Return a list of all node keys in this graph."""
        return sorted(self.nodes)

    @property
    def node_orders(self) -> dict:
        """Return how many connections a node has."""
        return {x: len(self.edge_options(x)) for x in self.nodes}

    @property
    def odd_nodes(self) -> List[int]:
        """Return a list of odd nodes only."""
        return [k for k in self.nodes if self.node_orders[k] % 2]

    def node_options(self, node: int) -> List[int]:
        """Returns an ascending list of (node, cost) tuples connected
        to this node."""
        options: List[int] = []
        for edge in self.edges.values():
            if edge.head == node:
                options.append(edge.tail)
            elif edge.tail == node:
                options.append(edge.head)
        return sorted(options)

    @property
    def is_eularian(self) -> bool:
        """Return True if all nodes are of even order."""
        return len(self.odd_nodes) == 0

    @property
    def is_semi_eularian(self) -> bool:
        """Return True if exactly 2 nodes are odd."""
        return len(self.odd_nodes) == 2

    @property
    def all_edges(self) -> List[Edge]:
        """Returns a list of all edges in this graph."""
        return list(self.edges.values())

    def find_edges(self, head: int, tail: int):
        """
        Returns a {key: edge} dictionary of all matching edges.
        """
        results: dict = {}
        for node, edge in self.edges.items():
            if (head, tail) == (edge.head, edge.tail) or (tail, head) == (
                edge.head,
                edge.tail,
            ):
                results[node] = edge
        return results

    def edge_options(self, node: int) -> Dict[int, Edge]:
        """Return dictionary of available edges for a given node."""
        return {k: v for k, v in self.edges.items() if node in (v.head, v.tail)}

    def edge_cost(self, head: int, tail: int):
        """Search for this edge."""
        weight: int = min(
            [
                edge.weight
                for edge in self.find_edges(head, tail).values()
                if edge.weight
            ]
        )
        return weight

    @property
    def total_cost(self) -> int:
        """Return the total cost of this graph."""
        return sum(x.weight for x in self.edges.values() if x.weight)

    def is_bridge(self, node) -> bool:
        """
        Return True if an edge is a bridge.

        Given an edge, utilize depth-first search to visit all
        connected nodes. If DFS reaches all unvisited nodes, then the given
        edge must not be a bridge.

        """
        graph: Graph = copy.deepcopy(self)

        start: int = graph.edges[node].tail  # Could start at either end.

        graph.remove_edge(node)  # Don't include the given edge

        stack: List[int] = []
        visited: Set[int] = set()  # Visited nodes
        while True:
            if start not in stack:
                stack.append(start)
            visited.add(start)
            nodes: List[int] = [
                x for x in graph.node_options(start) if x not in visited
            ]
            if nodes:
                start = nodes[0]  # Ascending
            else:  # Dead end
                try:
                    stack.pop()
                    start = stack[-1]  # Go back to the previous node
                except IndexError:  # We are back to the beginning
                    break

        if len(visited) == len(self.nodes):  # We visited every node
            return False  # ... therefore we did not disconnect the graph
        else:
            return True  # The edge is a bridge

    def __len__(self) -> int:
        return len(self.edges)

    def dump_graph(
        self,
        image_output_name: str = None,
        dot_output_name: str = None,
    ) -> None:
        """
        Dumps graph to image file (if `image_output_name` is not `None`) and
        dotfile (if `dot_output_name` is not None).
        """
        graph = MultiGraph(name="G")

        for edge in self.edges.values():
            graph.add_edge(
                edge.head,
                edge.tail,
                label=edge.weight,
            )

        graph.graph["node"] = {"shape": "circle"}

        if image_output_name:
            graph_plotted = to_agraph(graph)
            graph_plotted.layout("circo")
            graph_plotted.draw(image_output_name)

        if dot_output_name:
            write_dot(graph, dot_output_name)


class Edge:
    """A connection between nodes."""

    def __init__(self, head: int, tail: int, weight: int):
        self.head: int = head  # Start node
        self.tail: int = tail  # End node
        self.weight: int = weight  # aka Cost

    def __eq__(self, other: Edge) -> bool:
        return (self.head, self.tail, self.weight) == other

    def __ne__(self, other: Edge) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.contents)

    def __repr__(self) -> str:
        return "Edge({}, {}, {})".format(self.head, self.tail, self.weight)

    def __len__(self) -> int:
        """How many attribs we have. Kinda weird..."""
        return len([x for x in (self.head, self.tail, self.weight) if x is not None])

    def end(self, node: int) -> int:
        """Find the opposite end of this edge, given a node."""
        if node == self.head:
            return self.tail
        elif node == self.tail:
            return self.head
        else:
            raise ValueError("Node ({}) not in edge ({})".format(node, self))

    @property
    def contents(self) -> Tuple[int, int, int]:
        """A tuple containing edge contents."""
        return (self.head, self.tail, self.weight)
