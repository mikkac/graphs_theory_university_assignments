from networkx.drawing.nx_agraph import to_agraph, write_dot
from typing import List, Tuple, Dict

import numpy as np
import networkx as nx


class Graph:

    def __init__(self, graph_file_path: str) -> None:
        graph_as_str: str = self._read_graph_file(graph_file_path)
        self.vertices_num, self.edges_num = self._get_vertices_and_edges_num(
            graph_as_str)

        self.edges: List[Tuple[int, int]] = self._get_edges(graph_as_str)
        self.vertices: List[int] = self._get_vertices()

        self.adjacency_matrix: np.ndarray = self._get_adjacency_matrix()
        self.incidence_matrix: np.ndarray = self._get_incidence_matrix()

        self.degrees: List[int] = self._get_degrees()

        self.colors: List[int] = [-1] * self.vertices_num

        self.complement_edges: List[Tuple[int, int]
                                    ] = self._get_complement_edges()
        self.neighbour_vertices: List[Tuple[int,
                                            int]] = self._get_neighbour_vertices()

    def _read_graph_file(self, graph_file_path: str) -> str:
        """
        Reads graph from file associated with `graph_file_path`
        Graph file is expected to be in form of list of edges (lines 1-n):
        Line 0: <number of vertices> <number of edges>
        Line 1-n: <vertex 1> <vertex 2>
        ...
        """
        with open(graph_file_path) as f:
            lines = f.readlines()
        return lines

    def _get_vertices_and_edges_num(self, graph_as_str: str) -> Tuple[int, int]:
        """
        Fetches number of vertices and edges from read file
        """
        return tuple(int(x) for x in graph_as_str[0].split())

    def _get_vertices(self) -> List[int]:
        """
        Fetches unique labels of vertices from read file
        """
        return set([first for first, _ in self.edges] + [second for _, second in self.edges])

    def _get_edges(self, graph_as_str: str) -> List[Tuple[int, int]]:
        """
        Fetches list of edges from read file
        """
        return [tuple(int(x) for x in line.split()) for line in graph_as_str[1:]]

    def _get_edges_from_adjacency_matrix(self) -> List[Tuple[int, int]]:
        """
        Converts adjacency matrix to list of edges
        """
        edges: List[Tuple[int, int]] = []
        for first in range(self.adjacency_matrix.shape[0]):
            for second in range(first, self.adjacency_matrix.shape[1]):
                if first == second:
                    continue
                for _ in range(self.adjacency_matrix[first][second]):
                    edges.append((first+1, second+1))

        return edges

    def _get_adjacency_matrix(self) -> np.ndarray:
        """
        Prepares adjacency matrix
        """
        adjacency_matrix = np.zeros(
            (self.vertices_num, self.vertices_num), dtype=np.int8)

        for first, second in self.edges:
            adjacency_matrix[first - 1][second - 1] += 1
            adjacency_matrix[second - 1][first - 1] += 1

        return adjacency_matrix

    def _get_incidence_matrix(self) -> np.ndarray:
        """
        Prepares incidence matrix
        """
        incidence_matrix = np.zeros(
            (self.vertices_num, len(self.edges)), dtype=np.int8)
        for edge in range(1, len(self.edges) + 1):
            for vertex in range(1, self.vertices_num + 1):
                if vertex in self.edges[edge - 1]:
                    incidence_matrix[vertex - 1, edge - 1] += 1

        return incidence_matrix

    def _get_degrees(self) -> Dict[int, int]:
        """
        Returns degress of all vertices in the graph (key - vertex, value - degree)
        """
        return {
            vertex + 1: sum(self.adjacency_matrix[vertex])
            for vertex in range(self.adjacency_matrix.shape[0])
        }

    def _get_complement_edges(self) -> List[Tuple[int, int]]:
        """
        Returns list of complement edges
        """
        complement_edges: List[Tuple[int, int]] = []
        for first in range(self.adjacency_matrix.shape[0]):
            for second in range(first, self.adjacency_matrix.shape[1]):
                if first == second:
                    continue
                if self.adjacency_matrix[first, second] == 0:
                    complement_edges.append((first + 1, second + 1))

        return complement_edges

    def _get_neighbour_vertices(self) -> Dict[int, List[int]]:
        """
        Returns dict with key - vertex, value - neighbouring vertices of the vertex
        """
        neighbour_vertices: Dict[int, List[int]] = {
            vertex: [] for vertex in range(1, len(self.adjacency_matrix) + 1)
        }
        for first in range(self.adjacency_matrix.shape[0]):
            for second in range(self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[first, second] > 0:
                    neighbour_vertices[first + 1].append(second + 1)

        return neighbour_vertices

    def is_multigraph(self) -> bool:
        """
        Indicates whether the graph is multigraph (multi edges and loops)
        """
        return any(
            self.adjacency_matrix[idx][idx] != 0 for idx in range(self.vertices_num)
        ) or np.any(self.adjacency_matrix > 1)

    def dump_graph(self, image_output_name: str) -> None:
        """
        Dumps graph to image file
        """
        graph = nx.MultiGraph(name="G")

        for first in range(self.adjacency_matrix.shape[0]):
            for second in range(first, self.adjacency_matrix.shape[1]):
                if first == second:
                    continue
                for _ in range(self.adjacency_matrix[first, second]):
                    graph.add_edge(first+1, second+1)

        graph.graph['node'] = {'shape': 'circle'}

        if image_output_name:
            graph_plotted = to_agraph(graph)
            graph_plotted.layout('circo')
            graph_plotted.draw(image_output_name)


def load_and_describe_graph(graph_file: str) -> None:
    graph: Graph = Graph(graph_file)

    print(f"\nLiczba wierzcholkow (rzad) grafu G: {graph.vertices_num}")
    print(f"Zbior wierzcholkow grafu G: {graph.vertices}")

    print(f"\nLiczba krawedzi (moc) grafu G: {graph.edges_num}")
    print(f"Zbior krawedzi E grafu G: {graph.edges}")

    print(f"\nMacierz sasiedztwa A grafu G:\n{graph.adjacency_matrix}")
    print(f"\nMacierz incydencji M grafu G:\n{graph.incidence_matrix}")

    print("\nStopnie wierzcholkow grafu G:")
    print(
        "".join([f"deg({vertex}) = {degree}\n" for vertex,
                degree in graph.degrees.items()])
    )

    print(f"Ciag stopni grafu G: {sorted(graph.degrees.values())}")

    print(
        f"\nGraf G jest grafem {'ogolnym' if graph.is_multigraph() else 'prostym'}")

    print(
        f"\nGraf G {'nie' if graph.complement_edges else ''} jest grafem pelnym")

    complement_edges: List[Tuple[int, int]] = graph.complement_edges
    if complement_edges:
        print("\nKrawedzie dopelnienia grafu G:")
        print(
            "".join([f"{first}-{second}\n" for first, second in complement_edges]))

    print("Lista wierzcholkow grafu G:")
    print(
        "".join(
            [
                f"{vertex} -> {neighbours}\n"
                for vertex, neighbours in graph.neighbour_vertices.items()
            ]
        )
    )

    graph.dump_graph(f"{graph_file.split('.')[0]}.jpg")


def _main() -> None:
    load_and_describe_graph("graph1.txt")
    load_and_describe_graph("graph2.txt")


if __name__ == "__main__":
    _main()
