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


def solve_subset_sum_problem_floats(inputs: List[float], value: float):
    decimal_count = 2

    fractions = [Fraction(round(float(x), decimal_count)
                          ).limit_denominator() for x in inputs]
    value_fractional = Fraction(
        round(float(value), decimal_count)).limit_denominator()

    # calculate a common factor of all denominators of the inputs and the value so that
    # the fractions can be represented as integers
    common_factor = math.lcm(
        *[x.denominator for x in fractions + [value_fractional]])

    # calcluate the integer repreesntations of the fractional values
    integers = [int(fraction * common_factor) for fraction in fractions]
    value_integer = int(value_fractional * common_factor)

    # run the subset sum problem solver on the integer repreesntations
    viable_subset = dpss.find_subset(integers, value_integer, len(inputs))

    if len(viable_subset) > 1:
        print("MULTIPLE VIABLE SUBSETS")
    # pick the first subset if there is one, else empty list
    viable_subset = [] if len(viable_subset) == 0 else viable_subset[0]

    viable_subset_floats = [integers.index(x) for x in viable_subset]
    viable_subset_floats = [inputs[integers.index(x)] for x in viable_subset]

    # viable_subset_floats = [float(Fraction(numerator, common_factor)) for numerator in viable_subset]
    return viable_subset_floats


def _solve_subset_sum_problem_floats(inputs: List[float], value: float):
    """ Try to find a sub-multiset of items with a sum of value.
    """
    inputs = [round(x, PRECISION_EXP) for x in inputs]
    items = np.asarray(inputs)
    items = np.floor((items * PRECISION)).astype(int, casting="unsafe")
    value = int(value * PRECISION)

    result = dpss.find_subset(items, value, len(items))
    result = np.asarray(result)
    result = result.astype(float) / PRECISION
    return result


def is_approximate_in(value, items) -> bool:
    # does items contain an item that is approximately the same as value?
    return any([math.isclose(value, item) for item in items])


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


def solve_(instance: AttackInferenceProblem):
    graph = instance.framework.graph
    found_attacks = set()
    degrees = [degree for _, degree in graph.nodes(data="ground_truth_degree")]
    weights = [degree for _, degree in graph.nodes(data="weight")]

    lcm = math.lcm()

    for argument, data in graph.nodes(data=True):

        weight: float = data["weight"]
        degree: float = data["ground_truth_degree"]

        if weight == 0 and degree != 0 or weight != 0 and degree == 0:
            return False

        if weight != 0 and degree != weight:
            T = (weight - degree) / degree
            subset_sum_result = solve_subset_sum_problem_floats(degrees, T)
            if len(subset_sum_result) <= 0:
                return False

            found_attacks = found_attacks.union(
                [(b, a) for b, a in graph.edges() if
                 a == argument and is_approximate_in(graph.nodes[b]["ground_truth_degree"], subset_sum_result)])

    for a, b in found_attacks:
        instance.set_edge_status(a, b, True)

    instance.categorise()


if __name__ == "__main__":
    problem = random_weighted_acyclic_graph_problem(4, 0.1)
    print(problem.framework.graph.nodes(data=True))
    solve(problem)

