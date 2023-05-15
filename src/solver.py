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

PRECISION_EXP = 2
PRECISION = 10**PRECISION_EXP

def random_weighted_acyclic_graph_problem(num_nodes, density, max_denominator=10):

    # Create an empty directed graph
    original_graph = nx.gnp_random_graph(num_nodes,density, directed=True)
    framework = WeightedArgumentationFramework()
    
    # Add the nodes to the graph
    for i in range(num_nodes):
        w = Fraction(1,random.randint(1,10)) 
        # w = Fraction(random.random()).limit_denominator(max_denominator)
        framework.graph.add_node(i, weight=w)

    # Add the edges to the graph with random weights between 0 and 1
    for (u,v) in original_graph.edges():
        if u<v:
            framework.graph.add_edge(u, v)


    problem_instance = AttackInferenceProblem(framework, HCategoriser())
    return problem_instance


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

        # Find the target value of attacker's final degrees
        target = ((weight / degree - 1) * lcm).numerator
        print("INT DEGREES", list(int_degrees.values()))
        subset_sum_result = subsetsum.solutions(list(int_degrees.values()), target)
        if target != 0:
            print(subset_sum_result)
            for other_argument in next(subset_sum_result):
                found_attacks.append((argument, other_argument))

    for a, b in found_attacks:
        print(a, b)
        instance.set_edge_status(a, b, True)

    instance.categorise()


if __name__ == "__main__":
    problem = random_weighted_acyclic_graph_problem(4, 0.10)

    verifier = Verifier()

    print(problem.framework.graph.nodes(data=True))

    print([data for _, _, data in problem.framework.graph.edges(data="actual_edge")])

    print([data for _, _, data in problem.framework.graph.edges(data="predicted_edge")])
    print(verifier(problem))
    solve(problem)

    print([data for _, _, data in problem.framework.graph.edges(data="predicted_edge")])
    print(verifier(problem))


