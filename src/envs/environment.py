from typing import Optional, Tuple

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Dataset

from src.core import AttackInferenceProblem
from src.visualisation import Visualiser
from src.verifier import Verifier



class AttackInferenceEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_interval": 0.0005 }

    def __init__ (self, dataset:Dataset[AttackInferenceProblem], verifier=Verifier(), render_mode=None):

        self.render_mode = render_mode
        self._dataset = dataset

        # the sampler provides random indices into the dataset
        self._sampler = iter(RandomSampler(self._dataset)) 
        
        # the verifier is needed to determine when an episode is over, 
        # i.e. "good enough" solution has been found.
        self.verifier = verifier

        self._new_instance()

    def _get_obs(self):
        # return the observation derived from the current state
        return self._current_problem

    def _new_instance(self): 
        # move on to next problem instance
        self._current_problem: AttackInferenceProblem = self._dataset[next(self._sampler)]
        self._visualiser = Visualiser(self._current_problem, 
                                    render_interval=AttackInferenceEnvironment.metadata["render_interval"])
        
        # The action_space consist of all possible edges in the graph
        n_edges = self._current_problem.framework.graph.number_of_edges()

        self.action_space = spaces.Discrete(n_edges)
        self.observation_space = spaces.Discrete(1) # placeholder


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[AttackInferenceProblem, dict]:
        """ Reset the environment. Will randomly sample a new graph from the dataset.
            The action space is updated to reflect the new number of possible edges,
            since these might differ between different graphs in the same dataset.
        """
        super().reset(seed=seed, options=options)

        self._new_instance()

        return self._get_obs(), {}

    def step(self, action: int):
        """ Take an edge as an input. The edge will be "flipped" in the graph,
            i.e. go from predicted to not predicted or vice-versa 
        """
        self._current_problem.flip_edge(action)
        self._visualiser.flip_edge(self._current_problem.get_edge_from_index(action))

        reward = 0 

        self._current_problem.categorise() # need to update the predicted degrees
        terminated = self.verifier(self._current_problem)

        return self._current_problem, reward, terminated, False, {}

    def render(self):
        """ The behaviour of this method relies on the render_mode.
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
        """ This method is not needed. """
        pass


if __name__ == "__main__":
    environment = gym.make("AttackInference-v0")
 

