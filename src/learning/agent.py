from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from glob import glob
from random import random
import time
from typing import Any, Callable, Dict, Generic, List, Tuple, TypeVar, Optional

import numpy as np
import gymnasium as gym
import torch.nn.functional as F
import torch
from torch_geometric.data import Data

ReplayBufferItemType = TypeVar("ReplayBufferItemType")


class ReplayBuffer(deque, Generic[ReplayBufferItemType]):
    """Convenience wrapper extending a deque with random sampling capabilities."""

    def __init__(self, maxlen: int):
        super().__init__(maxlen=maxlen)

    def sample(
            self, batch_size: int, always_include_last: bool = True
    ) -> List[ReplayBufferItemType]:
        """
        Return 'batch_size'-number of elements, sampled at random from the replay buffer.
        If always_last ist true, the last element in the deque is al1ays included.
        """
        # pick indices randomly
        max_index = (
            len(self) - 1 if always_include_last else len(self)
        )  # don't randomly pick the last index twice

        if always_include_last:
            batch_size -= 1  # leave space for the last element

        indices = np.random.randint(max_index, size=(batch_size,))

        if always_include_last:
            indices = np.append(indices, [-1])

        # return a list of samples
        return [self[index] for index in indices]


class DeepQAgent:
    """Responsible for performing Q-learning based on Q-values provided by a GNN model."""

    ExperienceType = Tuple[Data, Data, int, int, bool]

    @dataclass
    class Settings:
        initial_epsilon: float = 0.9
        minimum_epsilon: float = 0.1
        epsilon_decay: float = 1e-4

        learning_rate: float = 1e-2  # alpha
        discount_rate: float = 0.5  # gamma

        # After reaching the specified length, the replay buffer will start to throw out old examples
        replay_buffer_length: int = 500
        batch_size: int = 50  # The number of experiences to sample for one batch

    def __init__(self, create_model: Callable[[], torch.nn.Module], settings: Settings = Settings()):
        self._settings = settings

        self._total_training_episodes = 0
        self._epsilon = self._settings.initial_epsilon

        self._online_model = (
            create_model()
        )  # The model that is making predictions during training
        self._offline_model = create_model()  # The model that is being trained

        self._replay_buffer: ReplayBuffer[DeepQAgent.ExperienceType] = ReplayBuffer(
            maxlen=self._settings.replay_buffer_length
        )

        self._optimiser = torch.optim.Adam(
            self._online_model.parameters(), lr=self._settings.learning_rate
        )

    def add_experience(self, experience: ExperienceType):
        """Add an experience without immediately learning from it."""
        self._replay_buffer.append(experience)

    def learn(self, experience: ExperienceType) -> Dict[str, Any]:
        """Experiences are (current_state, next_state, action, reward, done) tuples."""

        self.add_experience(experience)
        samples = self._replay_buffer.sample(self._settings.batch_size)

        # "transpose" to structure-of-arrays format
        current_states, next_states, actions, rewards, dones = zip(*samples)
        rewards = torch.tensor(rewards)
        actions = torch.tensor(actions)
        dones = torch.tensor(dones)

        # TODO: investigate the potential for batching
        # compute the q-values for the next states
        q_nexts = torch.stack([self._offline_model(state).squeeze() for state in next_states])

        # zero out all terminal q-values
        # note: ~ inverts a tensor of booleans, i.e True -> False and vice versa
        #       multiplying q_nexts with ~dones then makes all q-values 0 if 
        #       the next state is terminal.
        q_nexts = (q_nexts.T * (~dones)).T

        q_target = rewards + (
                self._settings.discount_rate * torch.max(q_nexts, dim=1).values
        )

        # get the q-values for actions in the current state
        q_currents = torch.stack(
            [self._offline_model(state).squeeze() for state in current_states]
        )

        # this selects the actually chosen actions from the q-values for all actions
        q_currents = torch.gather(q_currents, 1, actions.unsqueeze(dim=1)).squeeze()

        loss_function = torch.nn.SmoothL1Loss()
        # loss = F.smooth_l1_loss(q_target, q_currents)
        loss = loss_function(q_target, q_currents)

        self._optimiser.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_value_(self._online_model.parameters(), 100)

        self._optimiser.step()

        return {"loss": float(loss)}

    def get_action(self, data: Any, sample_random_function, do_exploration: bool = True) -> int:
        """Given the current state of the problem, produce the best action according to the current policy."""

        # Obtain node embeddings for current state, then readout edge values
        q_values = self._offline_model(data)

        # Check if exploration or exploitation should be performed (based on the epsilon)
        if do_exploration and random() < self._epsilon:
            action = sample_random_function()
            # action = np.random.randint(0, data.edge_index.shape[1])
        else:
            action = int(torch.argmax(q_values))

        return action

    def notify_episode_done(self):
        """ Update the epsilon value after the episode is done. """
        self._epsilon = np.max(
            [
                self._epsilon - self._settings.epsilon_decay,
                self._settings.minimum_epsilon,
            ]
        )

    @staticmethod
    def load_snapshot(
            model_name: str, filepath: Optional[str] = None
    ) -> Tuple["DeepQAgent", str]:
        """Load a snapshot from disk.
        If the filepath None, the latest snapshot will be loaded from the ./models/ folder,
        otherwise a string path is expected.
        """
        if filepath is None:
            filepath = sorted(glob(f"./models/{model_name}*.pt"))[-1]

        disk_data = torch.load(filepath)  # this is the raw data loaded from the disk

        # now, create a new agent that is then filled in with the data that was just loaded.
        agent = DeepQAgent()
        agent._online_model.load_state_dict(disk_data["gnn"])
        agent._epsilon = disk_data["epsilon"]
        agent._total_training_episodes = disk_data["total_training_episodes"]
        agent._settings = disk_data["settings"]
        return agent, filepath

    def save_snapshot(self, model_name: str, postfix: Optional[str] = None):
        """Save a snapshot of the model with the current parameters to the disk.
        The file will be stored in ./models/ and the name of the file is
        the current date + a unix timestamp with the fileending being ".pt".
        parameters:
            name: if none, the name of the file is just the timestamp scheme
                  otherwise, the name appears after the timestamp in the filename
        """

        # The filename will be a string containing the date in human readable format
        #   and also the unixtime to have a more exact differentiation for models created
        #   on the same day.
        datestring = datetime.now().strftime(f"%Y-%m-%d")
        unixtime = int(time.time())

        postfix = (
            f"-{postfix}" if postfix is not None else ""
        )  # if there is name, prefix it with a dash
        path = f"./models/{model_name}-{datestring}-{unixtime}{postfix}.pt"

        # Save the model parameters, current epsilon and settings
        torch.save(
            {
                "gnn": self._online_model.state_dict(),
                "epsilon": self._epsilon,
                "total_training_episodes": self._total_training_episodes,
                "settings": self._settings,
            },
            path,
        )


if __name__ == "__main__":
    class DQN(torch.nn.Module):
        def __init__(self, n_obs, n_action):
            super(DQN, self).__init__()
            self.layer1 = torch.nn.Linear(n_obs, 128)
            self.layer2 = torch.nn.Linear(128, 128)
            self.layer3 = torch.nn.Linear(128, n_action)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)


    env = gym.make("CartPole-v1")

    state, info = env.reset()
    state = torch.tensor(state)
    agent = DeepQAgent(lambda: DQN(len(state), env.action_space.n))

    print(state)
    EPISODES = 10

    def randomfn():
        return action

    for i in range(EPISODES):
        action = agent.get_action(state, sample_random_function=randomfn)
