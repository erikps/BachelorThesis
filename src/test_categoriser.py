
from core import WeightedArgumentationFramework
from categoriser import HCategoriser
import math


if __name__ == '__main__':
    framework = WeightedArgumentationFramework.from_file(
        "examples/example.apx").randomise_weights()

    categoriser = HCategoriser()
    categoriser(framework)

    nodes = framework.graph.nodes(data=True)
    def check(node, expected):
        degree = round(nodes[node]['acceptibility_degree'], 3)

        assert math.isclose(degree, expected), f"Acceptibility degree should be {expected}, but is {degree}"

    check('a', 0.421)
    check('b', 0.7)
    check('c', 0.437)
    check('d', 0.6)