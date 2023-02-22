import torch

from core import AttackInferenceProblem


def sum_of_squares(values: torch.Tensor):
        """ Return the sum of squares. """
        return values.square().sum()

class Verifier:
    def __init__ (self, tolerance=1e-09, reducer=sum_of_squares):
        self.tolerance = tolerance
        self.reducer = reducer
        
   
    def __call__ (self, problem: AttackInferenceProblem):
        """ Calculate the distance between the actual and the predicted degree. """

        errors = []
        for node, data in problem.framework.graph.nodes(data=True):
            error = data["predicted_degree"] - data["acceptibility_degree"]
            errors.append(error)

        return self.reducer(torch.tensor(errors))

