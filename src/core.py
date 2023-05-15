from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from fractions import Fraction
import pickle
import random
from numbers import Rational
import re
from typing import Tuple

import networkx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx


class AttackInferenceProblem:
    def __init__(
            self,
            framework: WeightedArgumentationFramework,
            categoriser: Categoriser,
            connect_fully=True,
            run_categoriser=True
    ):
        self.framework = framework
        self.categoriser = categoriser
        if run_categoriser:
            nodes = self.framework.graph.nodes()
            # Make the graph fully connected and add a flag for the actual and
            #   predicted existance of the edge
            for a in nodes:
                for b in nodes:
                    is_actual_edge = self.framework.graph.has_edge(
                        a, b
                    )  # does the edge actually exist?

                    if connect_fully or is_actual_edge:
                        self.framework.graph.add_edge(
                            a,
                            b,
                            actual_edge=is_actual_edge,
                            predicted_edge=False,
                            has_flipped=False,
                        )

            self.categoriser(
                self.framework, use_predicted_edges=False
            )  # This will create the ground-truth degree node attributes

            for node, data in self.framework.graph.nodes(data=True):
                self.framework.graph.add_node(
                    node, ground_truth_degree=data["predicted_degree"]
                )

            self.categorise()

    def to_fractional(self, max_denominator=100000):
        """ Turn the node features from floats into fractionals. """
        for node, data in self.framework.graph.nodes(data=True):
            for key in data.keys():
                data[key] = Fraction(
                    data[key]).limit_denominator(max_denominator)
                self.framework.graph.add_node(node, **data)

        for a, b, data in self.framework.graph.edges(data=True):
            for key in data.keys():
                data[key] = Fraction(
                    data[key]).limit_denominator(max_denominator)
                self.framework.graph.add_edge(a, b, **data)
        return self

    def reset_edges(self):
        """Revert all flip_edge calls in order to reset the problem to its original state."""
        for a, b in self.framework.graph.edges():
            self.framework.graph.add_edge(
                a, b, predicted_edge=False, has_flipped=False)

        self.categorise()

    def write_to_disk(self, filepath: str):
        """ Serialise the attack inference problem into a file on disk at the provided filepath. """
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def read_from_disk(filepath: str) -> "AttackInferenceProblem":
        """ Read the problem from disk at the filepath and return it. """
        with open(filepath, "rb") as file:
            return pickle.load(file)

    def get_edge_from_index(self, index: int) -> Tuple[str, str]:
        """Convert the edge from (node, node) format to index format."""
        return list(self.framework.graph.edges)[index]

    def set_edge_status(self, from_argument, to_argument, status: bool):
        self.framework.graph.add_edge(
            from_argument, to_argument, predicted_edge=status, has_flipped=True
        )

    def flip_edge(self, index_of_edge: int):
        """Change the "predicted_edge" edge attribute of the edge at index_of_edge
        from true to false or false to true.
        """
        from_node, to_node, predicted_edge = list(
            self.framework.graph.edges.data("predicted_edge")
        )[index_of_edge]
        self.framework.graph.add_edge(
            from_node, to_node, predicted_edge=not predicted_edge, has_flipped=True
        )

    def categorise(self):
        """Geneate the predicted acceptibility degrees for the current predicted attacks."""
        self.categoriser(self.framework)

    def as_true_graph(self):
        """ Return the argument graph with all "fake" edges removed. """
        copy = deepcopy(self.framework.graph)
        edges_to_remove = []
        for a, b, data in copy.edges(data=True):
            if not data["actual_edge"]:
                edges_to_remove.append((a, b))

        print(edges_to_remove)

        copy.remove_edges_from(edges_to_remove)
        return copy

    def to_torch_geometric(self) -> Data:
        """Convert an attack inference problem into the pytorch-geometric representation needed for learning."""

        # First need to specify which node and edge attributes to keep
        node_attributes = ["weight", "ground_truth_degree", "predicted_degree"]
        edge_attributes = ["predicted_edge", "has_flipped"]
        data: Data = from_networkx(
            self.framework.graph,
            group_node_attrs=node_attributes,
            group_edge_attrs=edge_attributes,
        )

        # data.y contains the information about which edges were in the ground-truth graph
        data.y = torch.Tensor(
            [int(x) for (_, _, x) in self.framework.graph.edges.data("actual_edge")]
        )

        data.edge_weight = data.edge_attr
        return data

    def randomise(self) -> "AttackInferenceProblem":
        """ Randomise the predicted edges. """
        edges = self.framework.graph.edges()
        values = [random.random() > 0.5 for _ in edges]
        edges = [(a, b, {"predicted_edge": value, "has_flipped": value})
                 for (a, b), value in zip(edges, values)]
        self.framework.graph.add_edges_from(edges)
        self.categorise()
        return self


class WeightedArgumentationFramework:
    """Abstract argumentation framework with weights for each node."""

    def __init__(self):
        self.graph: networkx.DiGraph = networkx.DiGraph()

    @staticmethod
    def from_file(filename: str):
        with open(filename) as file:
            content = "\n".join(file.readlines())
            return WeightedArgumentationFramework.from_string(content)

    def reset_predicted_degree(self):
        """Reset the predicted degree to the intial weight."""
        for node, weight in self.graph.nodes.data("weight"):
            self.graph.add_node(node, predicted_degree=weight)

    @staticmethod
    def from_string(string: str) -> "WeightedArgumentationFramework":
        ARGUMENT_REGEX = r"arg\(\s*(\w+)\s*\)"
        ATTACK_REGEX = r"att\(\s*(\w+)\s*,\s*(\w+)\s*\)"
        WEIGHT_REGEX = r"wgt\((\w)\s*,\s*(\d+\.\d+)\s*\)\."

        def byfirst(x):
            return x[0]

        weighted_argumentation_framework = WeightedArgumentationFramework()

        # arguments are names, e.g. "a", "b", ...
        arguments = sorted(re.findall(ARGUMENT_REGEX, string))

        # assign to each symbol an index used later for the ArgumentationFramework's internal representation of graphs
        label_mapping = {argument: index for index,
                         argument in enumerate(arguments)}

        # attacks are (symbol, symbol) tuples
        attacks = re.findall(ATTACK_REGEX, string)

        # intially weights are (index, float) tuples ...
        weights = sorted(
            [
                (label_mapping[argument], float(weight))
                for argument, weight in re.findall(WEIGHT_REGEX, string)
            ],
            key=byfirst,
        )

        # ... then transformed to just weights, ordered by the index
        weights = [weight for _, weight in weights]

        for argument, weight in zip(arguments, weights):
            weighted_argumentation_framework.graph.add_node(
                argument, weight=weight, predicted_degree=weight
            )
        weighted_argumentation_framework.graph.add_edges_from(attacks)
        return weighted_argumentation_framework

    def randomise_weights(self, randomiser=random.random):
        """Assign random weights based on the randomiser.
        randomiser: function with no parameters that returns a random value, defaults to random.random.
        """
        for node, data in self.graph.nodes(data=True):
            if "weight" in data and data["weight"] is not None:
                continue

            weight = randomiser()
            self.graph.add_node(node, weight=weight, predicted_degree=weight)
        return self


def random_fraction():
    return Fraction(random.random())


class Categoriser(ABC):
    """
    Abstract base class for all categorisers. A categoriser takes a weighted argumentation framework
    and returns a final acceptability degree.
    """

    @abstractmethod
    def __call__(
            self,
            weighted_argumentation_framework: WeightedArgumentationFramework,
            use_predicted_edges=True,
    ):
        """Apply the categoriser to the weighted argumentation framework provided"""
        pass
