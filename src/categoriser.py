from typing import Any, Dict, List, Tuple
import torch

from core import WeightedArgumentationFramework, Categoriser

edges = torch.tensor([[1, 0], [2, 0]])
weights = torch.tensor([1.0, 0.8, 0.5])


class HCategoriser(Categoriser):

    def __init__(self, stop_delta: float = 0.0000001):
        """
        stop_delta: if the mean squared delta of all degree changes is lesser than stop_delta,
                    stop computation and return
        """
        self.stop_delta = stop_delta

    def __call__(self, framework: WeightedArgumentationFramework):
        """
        Calculate the weighted h-categoriser weights by iteratively updating node weights based on attacking nodes' weights.
        Stop iteration once convergence occours, i.e. the mean absolute difference between node
        values being smaller than stop_delta.

        """

        mean_squared_delta = None
        number_of_nodes = framework.graph.number_of_nodes()

        while mean_squared_delta is None or mean_squared_delta > self.stop_delta:
            nodes: List[Tuple[str, Dict[str, Any]]
                        ] = framework.graph.nodes(data=True)
            for node, data in nodes:

                # first, accumulate all of the node's attacker's weights
                attackers = list(framework.graph.predecessors(node))

                sum_of_attack_degrees = sum(
                    [nodes[attacker]["acceptibility_degree"] for attacker in attackers])

                # this is the h-categoriser formula yielding the new acceptibility degree
                current_acceptibility_degree = data["weight"] / \
                    (sum_of_attack_degrees + 1)

                # add the new acceptibility degree as a seperate attribute so that
                # the nodes after the current one arent influenced by the new value
                framework.graph.add_node(
                    node, new_acceptibility_degree=current_acceptibility_degree)

            mean_squared_delta = 0  # reset last mean squared delta

            # Need to update the actual acceptibility degrees.
            for node, data in nodes:
                squared_delta = (
                    data["acceptibility_degree"] - data["new_acceptibility_degree"]) ** 2
                mean_squared_delta += squared_delta / number_of_nodes
                framework.graph.add_node(
                    node, acceptibility_degree=data["new_acceptibility_degree"])

        # clean up the new_acceptibility attribute on the nodes
        for _, data in framework.graph.nodes(data=True):
            del data["new_acceptibility_degree"]



class CardBasedCategoriser(Categoriser):
    # TODO: implement
    pass

