import torch

import math

from src.core import AttackInferenceProblem


def sum_of_squares(values: torch.Tensor):
        """ Return the sum of squares. """
        return values.square().sum()

class Verifier:
    def __init__ (self, tolerance=1e-09, reducer=sum_of_squares):
        self.tolerance = tolerance
        self.reducer = reducer

    
    def evaluate(self, problem: AttackInferenceProblem):
        errors = []
        for _, data in problem.framework.graph.nodes(data=True):
            error = data["predicted_degree"] - data["ground_truth_degree"]
            errors.append(error)
        return self.reducer(torch.tensor(errors))

   
    def __call__ (self, problem: AttackInferenceProblem):
        """ Calculate the distance between the actual and the predicted degree. """
        return math.isclose(self.evaluate(problem), 0, rel_tol=self.tolerance)

