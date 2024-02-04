import pickle
from graph import Graph, Part
from typing import List, Callable, Set, Tuple
from sklearn.model_selection import train_test_split
from node import Node

class Ordering:
    def __init__(self, y: List[Graph], degreeAggregate: Callable[[List[int]], int]):
        degrees = {}
        for graph in y:
            for (node, edges) in graph.get_edges().items():
                if node.get_part().get_part_id() not in degrees:
                    degrees[node.get_part().get_part_id()] = []
                degrees[node.get_part().get_part_id()].append(len(edges))
        self.keys = {part: degreeAggregate(degs) for (part, degs) in degrees.items()}

    def sort(self, x: Set[Part]) -> List[Part]:
        return sorted(x, key=lambda n: self.keys[n.get_part_id()], reverse=True)
    
    def get_compatible_graphs(self, graphs: List[Graph], train_rate=False):
        compatible_graphs = []
        for graph in graphs:
            compatible = True
            appendOrder = [0]
            edges = {node: edges.copy() for (node, edges) in graph.get_edges().items()}
            nodes: List[Node] = sorted(graph.get_nodes(), key=lambda n: self.keys[n.get_part().get_part_id()])
            for node in nodes[:-1]:
                if len(edges[node]) != 1:
                    if train_rate:
                        self.keys[node.get_part().get_part_id()] += train_rate
                    compatible = False
                    break
                parent = edges[node][0]
                appendOrder.append(len(nodes) - nodes.index(parent) - 1)
                edges[parent].remove(node)
            if compatible:
                part_seq = list(map(lambda node: int(node.get_part().get_part_id()), nodes))
                compatible_graphs.append((graph, part_seq[::-1], appendOrder[::-1]))
        if train_rate:
            print(f"Training on {len(graphs)} graphs: ", end="")
        print(f"{len(compatible_graphs) / len(graphs)} accuracy")
        return compatible_graphs

def get_splits():
    with open('data/graphs.dat', 'rb') as file:
        all_graphs: List[Graph] = pickle.load(file)
        X_train, X_temp, y_train, y_temp = train_test_split(list(map(lambda g: g.get_parts(), all_graphs)), all_graphs, test_size=0.3, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0) 

        splits = {"x_train": X_train, "y_train": y_train, "x_val": X_val, "x_test": X_test, "y_val": y_val, "y_test": y_test}
        return splits
    
def get_ordering(debug_accuracy=False):
    splits = get_splits()
    avgOrder = Ordering(splits["y_train"], lambda n: sum(n)/len(n))
    if debug_accuracy:
        avgOrder.get_compatible_graphs(splits["y_train"], train_rate=0.1)
        avgOrder.get_compatible_graphs(splits["y_train"], train_rate=0.1)
        print("Train Accuracy: ", end="")
        avgOrder_compatibleGraphs = avgOrder.get_compatible_graphs(splits["y_train"])
        print("Validation Accuracy: ", end="")
        avgOrder.get_compatible_graphs(splits["y_train"])
    return avgOrder