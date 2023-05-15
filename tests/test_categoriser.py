import unittest

from src.core import WeightedArgumentationFramework, AttackInferenceProblem, random_fraction
from src.categoriser import HCategoriser


class TestCategoriser(unittest.TestCase):
    def setUp(self):
        framework = WeightedArgumentationFramework.from_file(
            "./examples/test_example.apx"
        )

        categoriser = HCategoriser()
        problem = AttackInferenceProblem(framework, categoriser).to_fractional()

        self.nodes = problem.framework.graph.nodes(data=True)

    def check(self, node: str, expected: float):
        degree = round(self.nodes[node]["ground_truth_degree"], 2)
        self.assertAlmostEqual(
            float(degree),
            float(expected),
            msg=f"Acceptibility degree should be {expected}, but is {degree}",
        )

    def test_custom_case(self):
        self.check("a", 0.42)
        self.check("b", 0.7)
        self.check("c", 0.44)
        self.check("d", 0.6)


if __name__ == "__main__":
    unittest.main()
