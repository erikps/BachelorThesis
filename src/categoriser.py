from typing import Any, Dict, List, Tuple

import networkx
import torch

from src.core import WeightedArgumentationFramework, Categoriser

edges = torch.tensor([[1, 0], [2, 0]])
weights = torch.tensor([1.0, 0.8, 0.5])


class HCategoriser(Categoriser):
    def __init__(self, stop_delta: float = 0.00000000001):
        """
        stop_delta: if the mean squared delta of all degree changes is lesser than stop_delta,
                    stop computation and return
        """
        self.stop_delta = stop_delta

    def __call__(
        self, framework: WeightedArgumentationFramework, use_predicted_edges=True
    ):
        """
        Calculate the weighted h-categoriser weights by iteratively updating node weights based on attacking nodes' weights.
        Stop iteration once convergence occours, i.e. the mean absolute difference between node
        values being smaller than stop_delta.

        use_predicted_edges: if true, calculates the degree based on the predicted attacks.
                             otherwise, degree is based on the actual attacks (i.e. ground-truth).
        """

        mean_squared_delta = None
        number_of_nodes = framework.graph.number_of_nodes()

        # the predicted degree needs to be reset in case that changes to the degrees were made
        framework.reset_predicted_degree()

        while mean_squared_delta is None or mean_squared_delta > self.stop_delta:
            nodes: List[Tuple[str, Dict[str, Any]]] = framework.graph.nodes(data=True)

            predicted_edges = networkx.get_edge_attributes(
                framework.graph, "predicted_edge"
            )
            actual_edges = networkx.get_edge_attributes(framework.graph, "actual_edge")

            for node, data in nodes:

                def is_relevant_attacker(attacker):
                    # find out if the attacker is of the correct type
                    # i.e. predicted / actual depending on "use_predicted_edges" parameter

                    return (
                        predicted_edges[(attacker, node)]
                        if use_predicted_edges
                        else actual_edges[(attacker, node)]
                    )

                # first, accumulate all of the node's attacker's weights
                attacker_degrees = [
                    nodes[attacker]["predicted_degree"]
                    for attacker in framework.graph.predecessors(node)
                    if is_relevant_attacker(attacker)
                ]

                sum_of_attack_degrees = sum(attacker_degrees)

                # this is the h-categoriser formula yielding the new acceptibility degree
                current_acceptibility_degree = data["weight"] / (
                    sum_of_attack_degrees + 1
                )

                # add the new acceptibility degree as a seperate attribute so that
                # the nodes after the current one arent influenced by the new value
                framework.graph.add_node(
                    node, new_predicted_degree=current_acceptibility_degree
                )

            mean_squared_delta = 0  # reset last mean squared delta

            # Need to update the actual acceptibility degrees.
            for node, data in nodes:
                squared_delta = (
                    data["predicted_degree"] - data["new_predicted_degree"]
                ) ** 2
                mean_squared_delta += squared_delta / number_of_nodes
                framework.graph.add_node(
                    node, predicted_degree=data["new_predicted_degree"]
                )

        # clean up the new_acceptibility attribute on the nodes
        for _, data in framework.graph.nodes(data=True):
            del data["new_predicted_degree"]


class CardBasedCategoriser(Categoriser):
    # TODO: implement
    pass
