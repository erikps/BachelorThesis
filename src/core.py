from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import random
import re
from typing import List, Dict

import networkx
import torch


class AttackInferenceProblem:
    def __init__(self, framework: WeightedArgumentationFramework, categoriser: Categoriser):
        self.framework = framework.randomise_weights()
        self.categoriser = categoriser

        # Geneate desired acceptibility degrees.
        self.categoriser(framework)

        nodes = self.framework.graph.nodes()

        # Make the graph fully connected and add a flag for the actual and
        #   predicted existance of the edge
        for a in nodes:
            for b in nodes:
                is_actual_edge = self.framework.graph.has_edge(
                    a, b)  # does the edge actually exist?
                self.framework.graph.add_edge(
                    a, b, actual_edge=is_actual_edge, predicted_edge=False)

        networkx.set_node_attributes(
            self.framework.graph, 0, 'predicted_degree')


class WeightedArgumentationFramework:

    def __init__(self):
        self.graph: networkx.DiGraph = networkx.DiGraph()

    @staticmethod
    def from_file(filename: str):
        with open(filename) as file:
            content = "\n".join(file.readlines())
            return WeightedArgumentationFramework.from_string(content)

    @staticmethod
    def from_string(string: str) -> "WeightedArgumentationFramework":
        ARGUMENT_REGEX = r"arg\(\s*(\w+)\s*\)"
        ATTACK_REGEX = r"att\(\s*(\w+)\s*,\s*(\w+)\s*\)"
        WEIGHT_REGEX = r"wgt\((\w)\s*,\s*(\d+\.\d+)\s*\)\."

        def byfirst(x): return x[0]

        weighted_argumentation_framework = WeightedArgumentationFramework()

        # arguments are names, e.g. "a", "b", ...
        arguments = sorted(re.findall(ARGUMENT_REGEX, string))

        # assign to each symbol an index used later for the ArgumentationFramework's internal representation of graphs
        label_mapping = {argument: index for index,
                         argument in enumerate(arguments)}

        # attacks are (symbol, symbol) tuples
        attacks = re.findall(ATTACK_REGEX, string)

        # intially weights are (index, float) tuples ...
        weights = sorted([(label_mapping[argument], float(weight)) for argument, weight in re.findall(
            WEIGHT_REGEX, string)], key=byfirst)

        # ... then transformed to just weights, ordered by the index
        weights = [weight for _, weight in weights]

        for argument, weight in zip(arguments, weights):
            weighted_argumentation_framework.graph.add_node(
                argument, weight=weight, acceptibility_degree=weight)

        weighted_argumentation_framework.graph.add_edges_from(attacks)

        return weighted_argumentation_framework

    def randomise_weights(self, randomiser=random.random):
        """ Assign random weights based on the randomiser. 
            randomiser: function with no parameters that returns a random value, defaults to random.random.
        """
        for node, data in self.graph.nodes(data=True):
            if "weight" in data and data["weight"] is not None:
                continue

            weight = randomiser()
            self.graph.add_node(node, weight=weight,
                                acceptibility_degree=weight)

        return self


class Categoriser (ABC):

    """ 
    Abstract base class for all categorisers. A categoriser takes a weighted argumentation framework
    and returns a final acceptability degree. 
    """

    @abstractmethod
    def __call__(self, weighted_argumentation_framework: WeightedArgumentationFramework):
        """ Apply the categoriser to the weighted argumentation framework provided """
        pass
