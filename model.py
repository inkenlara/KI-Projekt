import pickle
from typing import List, Tuple, Dict, Set
from node import Node
from part import Part
from graph import Graph

from collections import defaultdict
import networkx as nx


# Sometimes we use hashable sets to represent edges
def edge(part1: Part, part2: Part) -> Set[int]:
    return frozenset({part1.get_part_id(), part2.get_part_id()})


# Count how often pairs of two parts appear together
def extract_pairs(graphs: List[Graph]) -> Dict[Set[Part], int]:
    pairs = defaultdict(int)
    for graph in graphs:
        for part1 in graph.get_parts():
            for part2 in graph.get_parts():
                pairs[edge(part1, part2)] += 1
    return pairs


# Get degrees of parts and count existing edges
def extract_edges(
    graphs: List[Graph],
) -> Tuple[Dict[Part, List[int]], Dict[Set[Part], int]]:
    degrees = defaultdict(list)
    edges = defaultdict(int)
    for graph in graphs:
        for node, neighbors in graph.get_edges().items():
            degrees[node.get_part().get_part_id()].append(len(neighbors))
            for neighbor in neighbors:
                edges[edge(node.get_part(), neighbor.get_part())] += 1
    return degrees, edges


# Edge priority before training is the ratio of edges to occurrences for a pair of parts
def init_edge_priority(
    pair_count: Dict[Set[Part], int], edge_count: Dict[Set[Part], int]
):
    edge_priority = defaultdict(float)
    for pair, count in pair_count.items():
        edge_priority[pair] = edge_count[pair] / count
    return edge_priority


# Check whether the graph described by edges is a subgraph of the target graph
def edges_are_subgraph(edges: List[Tuple[Part, Part]], target: Graph) -> bool:
    g = Graph()
    for edge0, edge1 in edges:
        g.add_undirected_edge(edge0, edge1)
    return nx.algorithms.isomorphism.GraphMatcher(
        target.to_nx(), g.to_nx(), node_match=lambda a, b: a == b
    ).subgraph_is_isomorphic()


# Find the first edge that is not part of the target graph
def first_wrong_edge(
    edges: List[Tuple[Part, Part]], target: Graph
) -> Tuple[Part, Part]:
    # Use binary search and edges_are_subgraphs to find the first wrong edge:
    min = 1
    max = len(edges)
    while min < max:
        mid = (min + max) // 2
        if edges_are_subgraph(edges[:mid], target):
            min = mid + 1
        else:
            max = mid
    return min - 1


# Return all neighbors of a part given edge list
def get_neighbor_ids(edges: List[Tuple[Part, Part]], part: Part) -> Set[int]:
    return frozenset(
        {t[0].get_part_id() for t in edges if part.equivalent(t[1])}.union(
            {t[1].get_part_id() for t in edges if part.equivalent(t[0])}
        )
    )


class InstanceBased:
    def __init__(
        self,
        y: List[Graph],
        order_train_rate=0.1,
        order_train_epochs=2,
        edge_train_rate=0.001,
        edge_train_epochs=10,
        edge_pickle_load=False,
        edge_pickle_store=False,
    ):
        # Extract information about training instances
        pairs_count = extract_pairs(y)
        degrees, edge_count = extract_edges(y)
        self.edge_priority = init_edge_priority(pairs_count, edge_count)
        self.avgDegrees = {part: sum(deg) / len(deg) for (part, deg) in degrees.items()}
        self.maxDegrees = {part: max(deg) for (part, deg) in degrees.items()}
        self.avgDegrees = defaultdict(float, self.avgDegrees)
        self.maxDegrees = defaultdict(int, self.maxDegrees)
        self.sort_priority = defaultdict(float, self.avgDegrees)
        self.train_order(y, order_train_rate, order_train_epochs)
        self.remaining_errors = []

        if edge_pickle_load:
            # Load pre-trained edge-priorities and known remaining-errors
            path = edge_pickle_load if isinstance(edge_pickle_load, str) else "data/"
            with open(f"{path}edge_priority.dat", "rb") as file:
                self.edge_priority = pickle.load(file)
            with open(f"{path}remaining_errors.dat", "rb") as file:
                self.remaining_errors = pickle.load(file)
        else:
            # Trains edge-priorities and stores remaining errors
            self.train_edges(y, edge_train_rate, edge_train_epochs)
        if edge_pickle_store:
            # Store edge-priorities and remaining errors for re-use
            path = edge_pickle_store if isinstance(edge_pickle_store, str) else "data/"
            with open(f"{path}edge_priority.dat", "wb") as file:
                pickle.dump(self.edge_priority, file)
            with open(f"{path}remaining_errors.dat", "wb") as file:
                pickle.dump(self.remaining_errors, file)

    # Sort parts to allow building tree sequentially
    def predict_order(self, x: Set[Part]) -> List[Part]:
        return sorted(x, key=lambda n: self.sort_priority[n.get_part_id()], reverse=1)

    # Train sorting to enable "dumbbell" graphs
    def train_order(self, graphs: List[Graph], train_rate, epochs):
        print(f"Possible accuracy when sequentially building a tree using sort method")
        for epoch in range(epochs):
            print(
                f"Train order, Epoch {epoch}: {100 * self.evaluate_order(graphs, train_rate) / len(graphs)}% accuracy"
            )
        print(
            f"Train order, Epoch {epochs}: {100 * self.evaluate_order(graphs) / len(graphs)}% accuracy"
        )

    # (Train and) evaluate supprt of graphs
    def evaluate_order(self, graphs: List[Graph], train_rate=False) -> int:
        compatible_graphs = 0
        for graph in graphs:
            compatible = True
            edges = {
                node: edges.copy() for (node, edges) in graph.get_edges().items()
            }  # Copy edges
            nodes: List[Node] = sorted(
                graph.get_nodes(),
                key=lambda n: self.sort_priority[n.get_part().get_part_id()],
            )  # Sort nodes reversed

            # Check whether leaves can be removed one by one using reversed sorting order
            for node in nodes[:-1]:
                if len(edges[node]) != 1:
                    # If not a leaf, it should've appeared later
                    compatible = False
                    if train_rate:
                        # If training is active, adapt priority accordingly
                        self.sort_priority[node.get_part().get_part_id()] += train_rate
                    break
                # Remove the leaf
                edges[edges[node][0]].remove(node)
            compatible_graphs += compatible
        return compatible_graphs

    # Return a correct edge for adding a new part given a target
    def _next_correct_edge(
        self, edges: List[Tuple[Part, Part]], part: Part, target: Graph
    ) -> Tuple[Part, Part]:
        parts = [edges[0][0]] + [edge[1] for edge in edges]
        candidates = sorted(
            [(p, part) for p in parts],
            key=lambda p: self.edge_priority[edge(p[0], p[1])],
            reverse=True,
        )
        for candidate in candidates:
            if edges_are_subgraph(edges + [candidate], target):
                return candidate

    # Check whether this situation is part of known errors during training
    def check_for_reminaing_errors(
        self, edges: List[Tuple[Part, Part]], best_edge: Tuple[Part, Part]
    ) -> Tuple[Part, Part]:
        # Build a dictionary to look up neighbors
        neighbors = defaultdict(set)
        for edge in edges:
            neighbors[edge[0].get_part_id()].add(edge[1].get_part_id())
            neighbors[edge[1].get_part_id()].add(edge[0].get_part_id())
        new_best_edge = best_edge
        max_occurrence = 0
        # Check whether a known error situation is occurring (including neighbors)
        for error in self.remaining_errors.keys():
            if (
                error[0] == best_edge[0].get_part_id()
                and error[1] == neighbors[best_edge[0].get_part_id()]
                and error[4] == best_edge[1].get_part_id()
                and error[2] in neighbors
                and error[3] == neighbors[error[2]]
            ):
                # From suitable error situations, the most frequent is chosen as alternative edge
                if self.remaining_errors[error] > max_occurrence:
                    max_occurrence = self.remaining_errors[error]
                    new_best_edge = (
                        (
                            [p for (p, _) in edges if p.get_part_id() == error[2]]
                            + [p for (_, p) in edges if p.get_part_id() == error[2]]
                        )[0],
                        best_edge[1],
                    )
        return new_best_edge

    # Sequentially builds graph after sorting parts
    def createGraph(
        self,
        unordered_parts: Set[Part],
        avgDegreeInfluence=0.05,
        maxDegreeInfluence=0.5,
        target=None,
        train_rate=None,
    ) -> Graph:
        # Order parts and add the first one to an empty graph
        parts = self.predict_order(unordered_parts)
        edges = [(parts[0], parts[1])]
        g = Graph()
        g.add_undirected_edge(parts[0], parts[1])
        # Add other parts one by one, with one edge each
        for i in range(2, len(parts)):
            best_edge = max(
                [(p, parts[i]) for p in parts[:i]],
                default=(parts[0], parts[i]),
                key=lambda p: self.edge_priority[edge(p[0], p[1])]
                + avgDegreeInfluence  # Prefer nodes that require more neighbors
                * (self.avgDegrees[p[0].get_part_id()] - g.get_degree(p[0]))
                - maxDegreeInfluence  # Dont exceed maxDegree
                * (g.get_degree(p[0]) == self.maxDegrees[p[0].get_part_id()]),
            )
            # Do not use known errors during training (would mess up training)
            if not target:
                best_edge = self.check_for_reminaing_errors(edges, best_edge)
            edges.append(best_edge)
            g.add_undirected_edge(best_edge[0], best_edge[1])
        if target and train_rate and g != target:
            # During training, adapt edge-priority and add incorrect choice to known errors
            first_wrong = first_wrong_edge(edges, target)
            correct = self._next_correct_edge(
                edges[:first_wrong], parts[first_wrong + 1], target
            )
            self.edge_priority[edge(correct[0], correct[1])] += train_rate
            self.remaining_errors[
                (
                    edges[first_wrong][0].get_part_id(),
                    get_neighbor_ids(edges[:first_wrong], edges[first_wrong][0]),
                    correct[0].get_part_id(),
                    get_neighbor_ids(edges[:first_wrong], correct[0]),
                    correct[1].get_part_id(),
                )
            ] += 1
            return None
        return g

    # Train for multiple epochs and only use remaining errors from the last one
    def train_edges(self, graphs: List[Graph], train_rate, epochs):
        print(f"Training edge priorities {epochs} epochs: ", end="")
        for epoch in range(epochs):
            self.remaining_errors = defaultdict(int)
            for graph in graphs:
                self.createGraph(graph.get_parts(), target=graph, train_rate=train_rate)
            print(f"{epoch + 1} ", end="")
        print("done")
