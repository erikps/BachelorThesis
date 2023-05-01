from fractions import Fraction
import math
import random
import time
from typing import List

import dpss
import subsetsum
import numpy as np
import networkx as nx
from src.categoriser import HCategoriser

from src.core import AttackInferenceProblem, WeightedArgumentationFramework
from src.dataset import AttackInferenceDataset
from src.verifier import Verifier


def solve(instance: AttackInferenceProblem):
    graph = instance.framework.graph
    found_attacks = list()

    # first need to find the lcm to represent numbers as integers
    degrees: List[int] = [degree for _,
                          degree in graph.nodes(data="ground_truth_degree")]
    weights: List[int] = [weight for _, weight in graph.nodes(data="weight")]

    lcm = math.lcm(*[x.denominator for x in degrees + weights])

    int_degrees = {
        name: (data["ground_truth_degree"]*lcm).numerator for name, data in graph.nodes(data=True)}

    for argument, data in graph.nodes(data=True):
        weight: float = data["weight"]
        degree: float = data["ground_truth_degree"]

        target = ((weight / degree - 1) * lcm).numerator
        subset_sum_result = subsetsum.solutions(list(int_degrees.values()), target)
        if target != 0:
            for other_argument in next(subset_sum_result):
                print(other_argument)
                found_attacks.append((argument, other_argument))

    for a, b in found_attacks:
        instance.set_edge_status(a, b, True)

    instance.categorise()

if __name__ == "__main__":
    problem = random_weighted_acyclic_graph_problem(4, 0.1)
    print(problem.framework.graph.nodes(data=True))
    solve(problem)

