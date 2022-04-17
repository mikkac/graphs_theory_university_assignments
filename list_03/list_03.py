""" Graphs theory assignment number 2 """
from __future__ import annotations

from queue import PriorityQueue
from typing import List, Tuple
from sys import maxsize
from itertools import permutations

from networkx.drawing.nx_agraph import to_agraph, write_dot
from networkx import MultiGraph
import numpy as np


class WeightedGraph:
    """
    Class which reads weighted graph from file, provides basic information about the graph,
    like vertices, edges, adjacency matrix, etc.
    """

    def __init__(self, graph_file_path: str) -> None:
        graph_as_str: str = self._read_graph_file(graph_file_path)
        self.vertices_num, self.edges_num = self._get_vertices_and_edges_num(
            graph_as_str
        )

        self.edges: List[Tuple[int, int, int]] = self._get_edges(graph_as_str)
        self.vertices: List[int] = self._get_vertices()

        self.adjacency_matrix: np.ndarray = self._get_adjacency_matrix()

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

    @staticmethod
    def _get_vertices_and_edges_num(graph_as_str: str) -> Tuple[int, int]:
        """
        Fetches number of vertices and edges from read file
        """
        return tuple(int(x) for x in graph_as_str[0].split())

    @staticmethod
    def _get_edges(graph_as_str: str) -> List[Tuple[int, int, int]]:
        """
        Fetches list of weighted edges from read file
        """
        return [tuple(int(x) for x in line.split()) for line in graph_as_str[1:]]

    def _get_vertices(self) -> List[int]:
        """
        Fetches unique labels of vertices from read file
        """
        return set(
            [first for first, _, _ in self.edges]
            + [second for _, second, _ in self.edges]
        )

    def _get_edges_from_adjacency_matrix(self) -> List[Tuple[int, int, int]]:
        """
        Converts adjacency matrix to list of edges
        """
        edges: List[Tuple[int, int, int]] = []
        for first in range(self.adjacency_matrix.shape[0]):
            for second in range(first, self.adjacency_matrix.shape[1]):
                if first == second:
                    continue
                for _ in range(self.adjacency_matrix[first][second]):
                    edges.append((first + 1, second + 1))

        return edges

    def _get_adjacency_matrix(self) -> np.ndarray:
        """
        Prepares adjacency matrix
        """
        adjacency_matrix: np.ndarray = np.empty(
            (self.vertices_num, self.vertices_num), dtype=object
        )

        for first, second, weight in self.edges:
            if adjacency_matrix[first - 1][second - 1] is None:
                adjacency_matrix[first - 1][second - 1] = []
            adjacency_matrix[first - 1][second - 1].append(weight)

            if adjacency_matrix[second - 1][first - 1] is None:
                adjacency_matrix[second - 1][first - 1] = []
            adjacency_matrix[second - 1][first - 1].append(weight)

        return adjacency_matrix

    def get_shortest_paths(self, start_vertex: int) -> List[float]:
        """Calculates shortest paths using Dijkstra algorithm"""
        start_vertex = start_vertex - 1
        if start_vertex < 0 or start_vertex > self.vertices_num:
            return []
        visited: List[int] = []
        costs: List[float] = [float("inf") for v in range(self.vertices_num)]
        costs[start_vertex] = 0

        queue = PriorityQueue()
        queue.put((0, start_vertex))

        while not queue.empty():
            (cost, current_vertex) = queue.get()
            visited.append(current_vertex)

            for neighbor in range(self.vertices_num):
                if self.adjacency_matrix[current_vertex][neighbor] is not None:
                    cost = min(self.adjacency_matrix[current_vertex][neighbor])
                    if neighbor not in visited:
                        old_cost = costs[neighbor]
                        new_cost = costs[current_vertex] + cost
                        if new_cost < old_cost:
                            queue.put((new_cost, neighbor))
                            costs[neighbor] = new_cost
        return costs

    def get_shortest_path(self, start_vertex: int, end_vertex: int) -> float:
        """Calculates shortest path using Dijkstra algorithm"""
        end_vertex = end_vertex - 1
        if start_vertex < 1 or start_vertex > self.vertices_num:
            return -1
        costs: List[float] = self.get_shortest_paths(start_vertex)
        if end_vertex >= len(costs):
            return -1
        return self.get_shortest_paths(start_vertex)[end_vertex]

    def naive_tsp(self, start_vertex: int) -> Tuple[float, List[int]]:
        """Solves Travelling Salesman Problem with naive approach"""
        start_vertex = start_vertex - 1
        if start_vertex < 0 or start_vertex > self.vertices_num:
            return -1
        vertices = [
            vertex for vertex in range(self.vertices_num) if vertex != start_vertex
        ]
        best_path: List[int] = None
        min_path = maxsize
        for permutation in permutations(vertices):
            visited_vertices: List[int] = [start_vertex + 1]
            current_pathweight = 0
            k = start_vertex
            for vertex in permutation:
                if self.adjacency_matrix[k][vertex] is not None:
                    current_pathweight += min(self.adjacency_matrix[k][vertex])
                    k = vertex
                    visited_vertices.append(k + 1)
            if self.adjacency_matrix[k][start_vertex] is not None:
                current_pathweight += min(self.adjacency_matrix[k][start_vertex])
                visited_vertices.append(start_vertex + 1)

            if (
                current_pathweight <= min_path
                and len(visited_vertices) == self.vertices_num + 1
            ):
                min_path = current_pathweight
                best_path = visited_vertices

        return min_path, best_path

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

        for first in range(self.adjacency_matrix.shape[0]):
            for second in range(first, self.adjacency_matrix.shape[1]):
                if first == second:
                    continue
                if self.adjacency_matrix[first, second] is not None:
                    for idx in range(len(self.adjacency_matrix[first, second])):
                        graph.add_edge(
                            first + 1,
                            second + 1,
                            label=self.adjacency_matrix[first, second][idx],
                        )

        graph.graph["node"] = {"shape": "circle"}

        if image_output_name:
            graph_plotted = to_agraph(graph)
            graph_plotted.layout("circo")
            graph_plotted.draw(image_output_name)

        if dot_output_name:
            write_dot(graph, dot_output_name)


def task_1(graph_file: str) -> None:
    """
    Checks whether graph is regular and if so, what degree does it have
    """
    graph: WeightedGraph = WeightedGraph(graph_file)

    print(f"\nZadanie 1 dla grafu wczytanego z {graph_file}:")

    graph.dump_graph(
        f"task_01_{graph_file.split('.')[0]}.jpg",
        f"task_01_{graph_file.split('.')[0]}.dot",
    )


def task_2(graph_file: str) -> None:
    """
    Displays shortest path between two vertices
    """
    graph: WeightedGraph = WeightedGraph(graph_file)

    print(f"\nZadanie 2 dla grafu wczytanego z {graph_file}:")

    start_vertex: int = 1
    end_vertex: int = 3
    cost = graph.get_shortest_path(start_vertex, end_vertex)
    print(
        f"Najkrotsza droga miedzy wierzcholkami {start_vertex} i {end_vertex} wynosi {cost}"
    )


def task_3(graph_file: str) -> None:
    """
    Displays shortest path between all possible pairs of vertices
    """
    graph: WeightedGraph = WeightedGraph(graph_file)

    print(f"\nZadanie 3 dla grafu wczytanego z {graph_file}:")

    for first in range(1, graph.vertices_num + 1):
        costs = graph.get_shortest_paths(first)
        for second in range(first + 1, graph.vertices_num + 1):
            if first == second:
                continue
            print(f"{first} - {second}\t{costs[second-1]}")


def task_4(graph_file: str) -> None:
    """
    Solve travelling salesman problem with naive approach
    """
    graph: WeightedGraph = WeightedGraph(graph_file)
    cost, best_path = graph.naive_tsp(1)

    cities = [
        "Warszawa",
        "Kraków",
        "Łódź",
        "Wrocław",
        "Poznań",
        "Gdańsk",
        "Szczecin",
        "Bydgoszcz",
        "Lublin",
        "Katowice",
    ]
    print(f"\nZadanie 4 dla grafu wczytanego z {graph_file}:")
    print(f"Rozwiązanie problemu komiwojazera dla podanej listy miast wynosi {cost} km")
    print("Optymalna kolejność węzłów:")
    for city in best_path:
        print(f"\t{cities[city-1]}")


def task_5(graph_file: str) -> None:
    """
    Solve travelling salesman problem with naive approach
    """
    graph: WeightedGraph = WeightedGraph(graph_file)

    print(f"\nZadanie 5 dla grafu wczytanego z {graph_file}:")
    for start_vertex in range(1, graph.vertices_num + 1):
        cost, best_path = graph.naive_tsp(start_vertex)
        print(
            f"Rozwiązanie problemu komiwojazera zaczynając od wierzchołka {start_vertex}\
                 wynosi {cost}, a najlepsza kolejność to {best_path}"
        )


def _main() -> None:
    task_1("g1.txt")
    task_1("g2.txt")

    task_2("g1.txt")
    task_2("g3.txt")

    task_3("g1.txt")

    task_4("cities.txt")

    task_5("g1.txt")


if __name__ == "__main__":
    _main()
