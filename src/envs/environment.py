from itertools import cycle
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data

from src.core import AttackInferenceProblem
from src.visualisation import Visualiser
from src.verifier import Verifier


def convert_to_torch_geometric_data(problem: AttackInferenceProblem) -> Data:
    """Convert an attack inference problem into the pytorch-geometric representation needed for learning."""

    # First need to specify which node and edge attributes to keep
    node_attributes = ["weight", "ground_truth_degree", "predicted_degree"]
    edge_attributes = ["predicted_edge"]
    data: Data = from_networkx(
        problem.framework.graph,
        group_node_attrs=node_attributes,
        group_edge_attrs=edge_attributes,
    )

    # data.y contains the information about which edges were in the ground-truth graph
    data.y = torch.Tensor(
        [int(x) for (_, _, x) in problem.framework.graph.edges.data("actual_edge")]
    )
    data.edge_weight = data.edge_attr

    return data


class AttackInferenceObservationSpace(gym.Space):
    def __init__(self, problem: AttackInferenceProblem):
        self._problem = problem

    def sample():
        """Sample a random observation"""


class AttackInferenceEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_interval": 0.0005}

    def __init__(
        self,
        dataset: Dataset[AttackInferenceProblem],
        verifier=Verifier(),
        render_mode=None,
    ):
        self.render_mode = render_mode
        self._dataset = dataset

        # the sampler provides random indices into the dataset
        self._sampler = cycle(RandomSampler(self._dataset))

        # the verifier is needed to determine when an episode is over,
        # i.e. "good enough" solution has been found.
        self.verifier = verifier

        self._new_instance()

    def _get_obs(self):
        # return the observation derived from the current state
        return self._current_problem

    def _new_instance(self):
        # move on to next problem instance
        self._current_problem: AttackInferenceProblem = self._dataset[
            next(self._sampler)
        ]
        self._visualiser = Visualiser(
            self._current_problem,
            render_interval=AttackInferenceEnvironment.metadata["render_interval"],
        )

        # The action_space consist of all possible edges in the graph
        n_edges = self._current_problem.framework.graph.number_of_edges()

        self.action_space = spaces.Discrete(n_edges)
        self.observation_space = spaces.Discrete(1)  # placeholder

    def reset(
        self,
        *,
        new_instance=True,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[AttackInferenceProblem, dict]:
        """Reset the environment. Will randomly sample a new graph from the dataset.
        The action space is updated to reflect the new number of possible edges,
        since these might differ between different graphs in the same dataset.
        """
        super().reset(seed=seed, options=options)

        # If a new instance is requested, generate it, otherwise reset the current instance
        if new_instance:
            self._new_instance()

        else:
            # this returns the problem to its original state
            self._current_problem.reset_edges()

        return self._get_obs(), {}

    def step(self, action: int):
        """Take an edge as an input. The edge will be "flipped" in the graph,
        i.e. go from predicted to not predicted or vice-versa
        """
        self._current_problem.flip_edge(action)
        self._visualiser.flip_edge(self._current_problem.get_edge_from_index(action))

        self._current_problem.categorise()  # need to update the predicted degrees
        terminated = self.verifier(self._current_problem)
        reward = 0 if terminated else -1

        return self._current_problem, reward, terminated, False, {}

    def render(self):
        """The behaviour of this method relies on the render_mode.
        - None:     nothing happens
        - "human":  a representation of the current state of the problem
                    instance is drawn in a matplotlib window
        """
        if self.render_mode is None:
            return

        elif self.render_mode == "human":
            error = self.verifier.evaluate(self._current_problem)
            self._visualiser.draw(error=error)

    def close(self):
        """This method is not needed for this environment."""
        pass


if __name__ == "__main__":
    environment = gym.make("AttackInference-v0")
