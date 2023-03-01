import unittest

from src.core import WeightedArgumentationFramework, AttackInferenceProblem
from src.categoriser import HCategoriser


class TestCategoriser(unittest.TestCase):

    def setUp(self):
        framework = WeightedArgumentationFramework.from_file(
            "./src/examples/example.apx").randomise_weights()

        categoriser = HCategoriser()

        problem = AttackInferenceProblem(framework, categoriser)

        self.nodes = problem.framework.graph.nodes(data=True)

    def check(self, node: str, expected: float):
        degree = round(self.nodes[node]['ground_truth_degree'], 3)
        self.assertAlmostEqual(degree, expected, msg=f"Acceptibility degree should be {expected}, but is {degree}")

    def test_custom_case(self):
        self.check('a', 0.421)
        self.check('b', 0.7)
        self.check('c', 0.437)
        self.check('d', 0.6)

if __name__ == '__main__':
    unittest.main()
   
