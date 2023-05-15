import numpy as np
import torch

import math

from src.core import AttackInferenceProblem


def sum_of_squares(values: np.ndarray):
    """Return the sum of squares."""
    return np.sum(values ** 2)


class Verifier:
    """ For a given argumentation framework, calculates if the predicted degrees are matching the ground truth. """

    def __init__(self, tolerance=1e-09, reducer=sum_of_squares):
        self.tolerance = tolerance
        self.reducer = reducer

    def evaluate(self, problem: AttackInferenceProblem):
        errors = []
        for _, data in problem.framework.graph.nodes(data=True):
            error = data["predicted_degree"] - data["ground_truth_degree"]
            errors.append(error)
        return self.reducer(np.asarray(errors))

    def __call__(self, problem: AttackInferenceProblem):
        """Calculate the distance between the actual and the predicted degree."""
        return math.isclose(self.evaluate(problem), 0, rel_tol=self.tolerance)
