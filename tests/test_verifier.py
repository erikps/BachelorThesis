import unittest

import networkx

from src.core import WeightedArgumentationFramework, AttackInferenceProblem
from src.categoriser import HCategoriser
from src.verifier import Verifier, sum_of_squares


def get_example_problem():
    """Example graph for testing purposes"""
    string = """
        arg(a).
        arg(b).
        arg(c).
        arg(d).
        att(b,a).
        att(c,a).
        att(d,c).
        wgt(a,0.9).
        wgt(b,0.7).
        wgt(c,0.7).
        wgt(d,0.6).
    """
    framework = WeightedArgumentationFramework.from_string(string)
    problem = AttackInferenceProblem(framework, HCategoriser())
    return problem


def set_predicted_equal_to_actual(problem: AttackInferenceProblem):
    for a, b, data in problem.framework.graph.edges(data=True):
        problem.framework.graph.add_edge(a, b, predicted_edge=data["actual_edge"])
    problem.categorise()  # need to update the predicted degrees


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = Verifier(reducer=sum_of_squares)

    def test_true_on_empty_graph(self):
        # trivial edge-case.
        categoriser = HCategoriser()
        framework = WeightedArgumentationFramework()
        problem = AttackInferenceProblem(framework, categoriser)
        self.assertTrue(self.verifier(problem))

    def test_example_with_no_predicted_attacks_is_zero_and_false(self):
        # if no attacks are predicted for the example graph, should be false
        example_problem = get_example_problem()
        self.assertFalse(self.verifier(example_problem))

    def test_example_with_predicted_equal_actual_attacks(self):
        # if the predicted attacks are the same as the actual attacks, the verifier should return True.
        example_problem = get_example_problem()
        set_predicted_equal_to_actual(example_problem)

        self.assertTrue(self.verifier(example_problem))

    def test_example_with_swapped_attacks(self):
        example_problem = get_example_problem()

        # Only change one attack
        set_predicted_equal_to_actual(example_problem)
        example_problem.framework.graph.add_edge("c", "a", predicted_edge=False)
        example_problem.categorise()

        # the values in the following have been pre-calculated
        # first, make sure that the nodes have the correct degrees
        def check(node: str, value: float):
            predicted = example_problem.framework.graph.nodes(data=True)[node][
                "predicted_degree"
            ]
            self.assertAlmostEqual(predicted, value)

        check("a", 0.9 / (1 + 0.7))
        check("b", 0.7)
        check("c", 0.7 / (1 + 0.6))
        check("d", 0.6)

        self.assertAlmostEqual(
            float(self.verifier.evaluate(example_problem)), 0.01175336072, places=4
        )
        self.assertFalse(self.verifier(example_problem))


if __name__ == "__main__":
    unittest.main()
