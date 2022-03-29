from __future__ import annotations

from enum import Enum
from networkx.drawing.nx_agraph import to_agraph, write_dot
from typing import List, Tuple, Dict, Optional

import numpy as np
import networkx as nx


class Graph:
    class EulerType(Enum):
        NONE = -1
        FULL = 0
        HALF = 1

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

    def _find_center_vertex(self) -> Optional[int]:
        """
        Finds center vertex in wheel graph
        """
        for vertex in range(self.vertices_num):
            if all((connection_vertex != vertex and self.adjacency_matrix[vertex][connection_vertex] == 1)
                   or (connection_vertex == vertex and self.adjacency_matrix[vertex][connection_vertex] == 0)
                   for connection_vertex in range(self.vertices_num)):
                return vertex
        return None

    def is_regular(self) -> bool:
        """
        Indicates whether graph is regular (`True') or not (`False`)
        Note: Graph is considered as regular if all degrees have the same value
        """
        init_degree: int = list(self.degrees.values())[0]
        return all(degree == init_degree for degree in self.degrees.values())

    def is_cyclic(self) -> bool:
        """
        Indicates whether graph is cyclic (`True') or not (`False`)
        Note: Graph is considered as cyclic if all degrees have value 2
        """
        return all(degree == 2 for degree in self.degrees)

    def is_wheel(self) -> bool:
        """
        Indicates whether graph is wheel (`True') or not (`False`)
        Note: Graph is considered as wheel if there is a center vertex which connects all other vertices
        and rest of vertices have degree 2
        """
        center_vertex: int = self._find_center_vertex()
        if center_vertex is None:
            return False

        for vertex in range(self.vertices_num):
            if vertex == center_vertex:
                continue
            if sum(self.adjacency_matrix[vertex]) != 3:
                return False

        return True

    def is_euler(self) -> Graph.EulerType:
        """
        Indicates whether graph is euler (`Graph.EulerType.FULL'), half euler (`Graph.EulerType.HALF`)
        or not euler in any way (`Graph.EulerType.NONE`)
        Note: Graph is considered as euler if all degrees have even value, half euler if exactly two
        vertices have odd degree
        """
        if all(degree % 2 == 0 for degree in self.degrees.values()):
            return Graph.EulerType.FULL

        if sum(degree % 2 == 1 for degree in self.degrees.values()) == 2:
            return Graph.EulerType.HALF

        return self.EulerType.NONE

    def convert_cyclic_to_wheel(self) -> bool:
        """
        Converts cyclic graph to wheel one
        """
        # Vertex "0" will become a central one
        central_vertex = 0
        if not self.is_cyclic():
            return False

        # Find vertices that were connected with a vertex that became a center
        vertices_to_connect = np.where(
            self.adjacency_matrix[central_vertex] == 1)[0]

        # Once vertex "0" is moved to the center, vertices that were connected
        # with it, have to connect with each other
        self.adjacency_matrix[vertices_to_connect[0]
                              ][vertices_to_connect[1]] = 1
        self.adjacency_matrix[vertices_to_connect[1]
                              ][vertices_to_connect[0]] = 1

        # Now connect rest of the vertices with the center one
        for vertex_idx in range(len(self.adjacency_matrix[central_vertex])):
            self.adjacency_matrix[central_vertex][vertex_idx] = 1
            self.adjacency_matrix[vertex_idx][central_vertex] = 1

        # Update edges and degrees
        self.edges = self._get_edges_from_adjacency_matrix()
        self.degrees = self._get_degrees()

        return True

    def convert_wheel_to_cyclic(self) -> bool:
        """
        Converts wheel graph to cyclic one
        """
        if not self.is_wheel():
            return False

        # Find center vertex
        center_vertex: int = self._find_center_vertex()

        # Pick vertices that will be connected after center vertex is removed
        vertex_to_connect_1: int = None
        vertex_to_connect_2: int = None
        if center_vertex == 0:
            vertex_to_connect_1 = 1
            vertex_to_connect_2 = 2
        elif center_vertex == self.vertices_num - 1:
            vertex_to_connect_1 = self.vertices_num - 2
            vertex_to_connect_2 = self.vertices_num - 3
        else:
            vertex_to_connect_1 = center_vertex + 1
            vertex_to_connect_2 = center_vertex - 1

        # Disconnect the vertices
        self.adjacency_matrix[vertex_to_connect_1][vertex_to_connect_2] = 0
        self.adjacency_matrix[vertex_to_connect_2][vertex_to_connect_1] = 0

        # Break all connections between center vertex and rest of vertices (except vertices to connect)
        for vertex in range(self.vertices_num):
            if vertex == center_vertex:
                for connection in range(self.vertices_num):
                    if connection not in (vertex_to_connect_1, vertex_to_connect_2):
                        self.adjacency_matrix[vertex][connection] = 0
            if vertex not in (vertex_to_connect_1, vertex_to_connect_2):
                self.adjacency_matrix[vertex][center_vertex] = 0

        # Update edges and degrees
        self.edges = self._get_edges_from_adjacency_matrix()
        self.degrees = self._get_degrees()

        return True

    def color_greedy(self):
        """
        Colors graph with greedy algorithm
        """
        self.colors = [-1] * self.vertices_num
        # Assign the first color to first vertex
        self.colors[0] = 0

        # A temporary array to store used colors.
        # True value of used_colors[cr] would mean that the
        # color cr is assigned to one of its adjacent vertices
        used_colors: List[bool] = [False] * self.vertices_num

        # Assign colors to remaining self.vertices_num-1 vertices
        for vertex in range(1, self.vertices_num):
            # Process all adjacent vertices and
            # flag their colors as unavailable
            for connection in range(self.vertices_num):
                if self.adjacency_matrix[vertex][connection] == 1 and self.colors[connection] != -1:
                    used_colors[self.colors[connection]] = True

            # Color vertex with first available color
            for color_index in range(self.vertices_num):
                if not used_colors[color_index]:
                    self.colors[vertex] = color_index
                    used_colors[color_index] = True
                    break

            # Reset the values back to false for the next iteration
            for connection in range(self.vertices_num):
                if self.colors[connection] != -1:
                    used_colors[self.colors[connection]] = False

    def dump_graph(self, image_output_name: str = None, dot_output_name: str = None, colors_as_labels: bool = False) -> None:
        """
        Dumps graph to image file (if `image_output_name` is not `None`), to dotfile (if `dot_output_name` is not None)
        If `colors_as_labels` is set to True the instead of labels, colors are used 
        """
        graph = nx.MultiGraph(name="G")

        if colors_as_labels:
            for vertex, color in enumerate(self.colors):
                graph.add_node(vertex+1, label=f"k{color}")

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

        if dot_output_name:
            write_dot(graph, dot_output_name)


def task_1(graph_file: str) -> None:
    """
    Checks whether graph is regular and if so, what degree does it have
    """
    graph: Graph = Graph(graph_file)

    print(f"\nZadanie 1 dla grafu wczytanego z {graph_file}:")
    if graph.is_regular():
        print(
            f"Graf jest grafem regularnym, stopnia {list(graph.degrees.values())[0]}")
    else:
        print(f"Graf nie jest grafem regularnym")

    graph.dump_graph(f"task_01_{graph_file.split('.')[0]}.jpg")


def task_2(graph_file: str) -> None:
    """
    Checks if graph is cyclic and if so, converts it to wheel one
    """
    graph: Graph = Graph(graph_file)

    cyclic: bool = graph.is_cyclic()

    print(f"\nZadanie 2 dla grafu wczytanego z {graph_file}:")
    print(f"Graf {'' if cyclic else 'nie'} jest grafem cyklicznym")

    if cyclic:
        print(
            f"\nMacierz sasiedztwa grafu G przed konwersją:\n{graph.adjacency_matrix}")

        graph.convert_cyclic_to_wheel()

        print(
            f"\nMacierz sasiedztwa grafu G po konwersji na graf kołowy:\n{graph.adjacency_matrix}")
        print(f"\nLista krawędzi po konwersji:")
        print("".join([f"{edge[0]} - {edge[1]}\n" for edge in graph.edges]))

        graph.dump_graph(f"task_02_{graph_file.split('.')[0]}_to_wheel.jpg",
                         f"task_02_{graph_file.split('.')[0]}_to_wheel.dot")
    else:
        graph.dump_graph(f"task_02_{graph_file.split('.')[0]}.jpg")


def task_3(graph_file: str) -> None:
    """
    Checks if graph is wheel and if so, converts it to cyclic one
    """
    graph: Graph = Graph(graph_file)

    wheel: bool = graph.is_wheel()

    print(f"\nZadanie 3 dla grafu wczytanego z {graph_file}:")
    print(f"Graf {'' if wheel else 'nie'} jest grafem kołowym")

    if wheel:
        print(
            f"\nMacierz sasiedztwa grafu G przed konwersją:\n{graph.adjacency_matrix}")

        graph.convert_wheel_to_cyclic()

        print(
            f"\nMacierz sasiedztwa grafu G po konwersji na graf kołowy:\n{graph.adjacency_matrix}")
        print(f"\nLista krawędzi po konwersji:")
        print("".join([f"{edge[0]} - {edge[1]}\n" for edge in graph.edges]))

        graph.dump_graph(f"task_03_{graph_file.split('.')[0]}_to_cyclic.jpg",
                         f"task_03_{graph_file.split('.')[0]}_to_cyclic.dot")

    else:
        graph.dump_graph(f"task_03_{graph_file.split('.')[0]}.jpg")


def task_4(graph_file: str) -> None:
    """
    Colors graph with greedy algorithm 
    """
    graph: Graph = Graph(graph_file)
    graph.color_greedy()

    print(f"\nZadanie 4 dla grafu wczytanego z {graph_file}:")
    print(f"Kolory przypisane do wierzchołków:")
    for vertex, color in enumerate(graph.colors):
        print(f"{vertex+1} - k{color}")
    graph.dump_graph(
        f"task_04_{graph_file.split('.')[0]}.jpg", f"task_04_{graph_file.split('.')[0]}.dot", True)


def task_5(graph_file: str) -> None:
    """
    Checks whether graph is euler, half-euler or not euler type at all
    """
    graph: Graph = Graph(graph_file)

    print(f"\nZadanie 5 dla grafu wczytanego z {graph_file}:")
    euler_type: Graph.EulerType = graph.is_euler()

    if euler_type == Graph.EulerType.FULL:
        print("Graf jest grafem eulerowskim (stopień każdego wierzchołka jest liczbą parzystą)")
    elif euler_type == Graph.EulerType.HALF:
        print("Graf jest grafem półeulerowskim (stopień dokładnie dwóch wierzchołków jest liczbą nieparzystą)")
    else:
        print("Graf nie jest ani grafem eulerowskim, ani półeulerowskim")

    graph.dump_graph(f"task_05_{graph_file.split('.')[0]}.jpg")


def _main() -> None:
    task_1("regular.txt")
    task_1("not_regular.txt")

    task_2("not_regular.txt")
    task_2("cyclic.txt")

    task_3("not_regular.txt")
    task_3("cyclic.txt")
    task_3("wheel.txt")

    task_4("cyclic.txt")
    task_4("wheel.txt")
    task_4("regular.txt")

    task_5("euler.txt")
    task_5("half_euler.txt")
    task_5("not_euler.txt")


if __name__ == "__main__":
    _main()
