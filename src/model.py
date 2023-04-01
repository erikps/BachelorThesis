from collections import deque
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from random import random
import time
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Optional

import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import RandomSampler
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, AGNNConv
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from src.core import AttackInferenceProblem
from src.dataset import AttackInferenceDataset
from src.envs.environment import AttackInferenceEnvironment


class NeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        input_layer_size: int,
        output_layer_size: int,
        hidden_layer_count: int,
        hidden_layer_size: int,
        activation=torch.nn.ReLU,
    ):
        super().__init__()

        layers = [torch.nn.Linear(input_layer_size, hidden_layer_size), activation()]

        for _ in range(hidden_layer_count):
            layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(activation())
        layers.append(torch.nn.Linear(hidden_layer_size, output_layer_size))
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Apply all layers sequentially."""
        return self.sequential(x)


class GraphNeuralNetwork(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        number_edge_features: int,
        number_message_layers: int = 1,
        message_layer_size = 100
    ):

        super().__init__(aggr="add")
        message_input_size = number_edge_features + out_channels * 2
        self._message_network = NeuralNetwork(message_input_size, out_channels, number_message_layers, message_layer_size)

    def message(self, node_embeddings_i: Tensor, node_embeddings_j: Tensor, edge_features: Tensor):
        """ Create the message for a given node/node. """
        message = self._message_network(torch.cat((node_embeddings_i, node_embeddings_j, edge_features)))
        return message

    def forward(self, x, edge_features, edge_index):
        """ Start propagation """


    


class ReadoutModel(torch.nn.Module):
    """Responsble for computing node representations and reading out q-values from them.
    parameters:
        channel_count:              number of learned embeddings for each node
        readout_hidden_layer_count: number of hidden layers for the readout network

    """

    def __init__(self, channel_count=16, readout_hidden_layer_count=1) -> None:
        super().__init__()
        number_of_node_features = 3  # weight, ground_truth_degree and predicted_degree
        number_of_edge_features = 2  # predicted_edge, has_flipped

        # Set up the GCN layers
        self.conv1 = GCNConv(number_of_node_features, channel_count)
        self.conv2 = GCNConv(channel_count, channel_count)

        # self.conv1 = AGNNConv()
        # self.conv2 = AGNNConv()

        # The readout network takes a (attacker, attacked) pair and returns a q-value for it
        self.readout_network = NeuralNetwork(
            input_layer_size=2 * channel_count + number_of_edge_features,
            output_layer_size=1,
            hidden_layer_size=channel_count,
            hidden_layer_count=readout_hidden_layer_count,
        )

    def readout(
        self, node_embeddings: torch.Tensor, edge_index: torch.Tensor, edge_features
    ) -> torch.Tensor:
        """Use the readout network to determine the edge outputs
        parameters:
            node_embeddings: These are the node embeddings (channels) computed by the GNN of shape [node_count, channel_count]
            edge_index:      Tensor containing [0] the attackers and [1] the attacked for each attack in the graph
            edge_features:   The features associated with attacks, for example if it is supposed to exist
        """

        # first, separate the attacker and attacked nodes from the edge_index.
        attackers, attacked = (
            node_embeddings[edge_index[0]],
            node_embeddings[edge_index[1]],
        )

        # readout_input is of shape [edge_count, channel_count * 2 + edge_feature_count]
        readout_input = torch.cat((attackers, attacked, edge_features), dim=1)

        return F.relu(self.readout_network(readout_input))

    def forward(self, data):
        """Run both of the model's convolutional layers with an activation layer inbetween."""
        x = data.x
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return self.conv2(x, edge_index)


ReplayBufferItemType = TypeVar("ReplayBufferItemType")


class ReplayBuffer(deque, Generic[ReplayBufferItemType]):
    """Convienience wrapper extending a deque with random sampling capabilities."""

    def __init__(self, maxlen: int):
        super().__init__(maxlen=maxlen)

    def sample(self, batch_size: int) -> List[ReplayBufferItemType]:
        """Return 'batch_size'-number of elements, sampled at random from the replay buffer."""
        # pick indices randomly
        indices = torch.randint(len(self), size=(batch_size,))

        # return a list of samples
        return [self[index] for index in indices]


class DeepQAgent:
    """Responsible for performing Q-learning based on Q-values provided by a GNN model."""

    ExperienceType = Tuple[Data, Data, int, int, bool]

    @dataclass
    class Settings:
        minimum_epsilon: float = 0.1
        epsilon_decay: float = 0.001

        learning_rate: float = 1e-5  # alpha
        discount_rate: float = 0.99  # gamma

        # After reaching the specified length, the replay buffer will start to throw out old examples
        replay_buffer_length: int = 500
        batch_size: int = 50  # The number of experiences to sample for one batch

    def __init__(self, settings: Settings = Settings()):
        self._epsilon = 1.0
        self._settings = settings

        self._online_model = (
            ReadoutModel()
        )  # The model that is making predictions during training
        self._offline_model = ReadoutModel()  # The model that is being trained
        self._replay_buffer: ReplayBuffer[DeepQAgent.ExperienceType] = ReplayBuffer(
            maxlen=self._settings.replay_buffer_length
        )

        self._optimiser = torch.optim.Adam(
            self._online_model.parameters(), lr=self._settings.learning_rate
        )

    def add_experience(self, experience: ExperienceType):
        """Add an experience without immediatly learning from it."""
        self._replay_buffer.append(experience)

    def learn(self, experience: ExperienceType) -> Dict[str, Any]:
        """Experiences are (current_state, next_state, action, reward, done) tuples."""

        self.add_experience(experience)
        samples = self._replay_buffer.sample(self._settings.batch_size)

        # "transponse" to structure-of-arrays type data
        current_states, next_states, actions, rewards, dones = zip(*samples)
        rewards = torch.tensor(rewards)
        actions = torch.tensor(actions)

        # TODO: investigate the potential for batching
        # compute the q-values for the next states
        q_nexts = torch.stack([self._readout(state).squeeze() for state in next_states])
        q_nexts[dones] = 0.0

        q_target = rewards + (
            self._settings.discount_rate * torch.max(q_nexts, dim=1).values
        )

        # get the q-values for actions in the current state
        q_currents = torch.stack(
            [self._readout(state).squeeze() for state in current_states]
        )

        # # one-hot encode the actions
        # num_classes = q_currents.shape[1]
        # actions_one_hot: torch.Tensor = F.one_hot(
        #     actions, num_classes=num_classes
        # ).type(torch.FloatTensor)

        # # get the q-values of the actually selected actions
        # q_currents = torch.matmul(q_currents, actions_one_hot.T)

        # q_currents = q_currents.sum(dim=0)

        q_currents = torch.stack(
            [q_current[index] for q_current, index in zip(q_currents, actions)]
        )
        print(q_currents)

        loss = F.huber_loss(q_target, q_currents)
        loss.backward()

        self._optimiser.step()

        return {"loss": float(loss)}

    def _readout(self, data: Data):
        # run the gnn and perform readout in one step
        node_embeddings = self._online_model(data)
        result = self._online_model.readout(
            node_embeddings, data.edge_index, data.edge_attr
        )
        return result

    def get_action(self, data: Data) -> int:
        """Given the current state of the problem, produce the best action according to the current policy."""

        # Obtain node embeddings for current state, then readout edge values
        edge_values = self._readout(data)

        # Check if exploration or exploitation should be performed (based on the epsilon)
        if random() < self._epsilon:
            action = np.random.randint(0, data.edge_index.shape[1])
        else:
            action = int(torch.argmax(edge_values))

        # Update epsilon value
        self._epsilon = np.max(
            [
                self._epsilon - self._settings.epsilon_decay,
                self._settings.minimum_epsilon,
            ]
        )
        return action

    @staticmethod
    def load_snapshot(filepath: Optional[str] = None) -> Tuple["DeepQAgent", str]:
        """Load a snapshot from disk.
        If the filepath None, the latest snapshot will be loaded from the ./models/ folder,
        otherwise a string path is expected.
        """
        if filepath is None:
            filepath = sorted(glob("./models/*.pt"))[-1]

        disk_data = torch.load(filepath)  # this is the raw data loaded from the disk

        # now, create a new agent that is then filled in with the data that was just loaded.
        agent = DeepQAgent()
        agent._online_model.load_state_dict(disk_data["gnn"])
        agent._epsilon = disk_data["epsilon"]
        agent._settings = disk_data["settings"]
        return agent, filepath

    def save_snapshot(self):
        """Save a snapshot of the model with the current parameters to the disk.
        The file will be stored in ./models/ and the name of the file is
        the current date + a unix timestamp with the fileending being ".pt".
        """

        # The filename will be a string containing the date in human readable format
        #   and also the unixtime to have a more exact differentiation for models created
        #   on the same day.
        datestring = datetime.now().strftime(f"%Y-%m-%d")
        unixtime = int(time.time())
        path = f"./models/{datestring}-{unixtime}.pt"

        # Save the model parameters, current epsilon and settings
        torch.save(
            {
                "gnn": self._online_model.state_dict(),
                "epsilon": self._epsilon,
                "settings": self._settings,
            },
            path,
        )


def convert_to_torch_geometric_data(problem: AttackInferenceProblem) -> Data:
    """Convert an attack inference problem into the pytorch-geometric representation needed for learning."""

    # First need to specify which node and edge attributes to keep
    node_attributes = ["weight", "ground_truth_degree", "predicted_degree"]
    edge_attributes = ["predicted_edge", "has_flipped"]
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


@dataclass
class TrainingSettings:
    # number of actions taken between learning from the experience buffer
    learn_interval = 1


def train_agent(
    agent: DeepQAgent = DeepQAgent(),
    environment: AttackInferenceEnvironment = AttackInferenceEnvironment(
        AttackInferenceDataset.example_dataset(), render_mode="human"
    ),
    episodes=10,
):
    """Train the agent using the provided environment."""

    def prefill_experience_buffer():
        # fill the experience buffer with random experiences
        old_observation, _ = environment.reset(new_instance=False)
        old_data = convert_to_torch_geometric_data(old_observation)

        for _ in range(agent._replay_buffer.maxlen):
            action = environment.action_space.sample()
            new_observation, reward, done, _, _ = environment.step(action)
            new_data = convert_to_torch_geometric_data(new_observation)
            agent.add_experience((old_data, new_data, action, reward, done))
            old_data = new_data

        environment.reset(new_instance=False)

    for episode in range(episodes):
        prefill_experience_buffer()
        old_observation, _ = environment.reset()
        old_data = convert_to_torch_geometric_data(old_observation)

        done = False
        while not done:
            action = agent.get_action(old_data)
            new_observation, reward, done, _, _ = environment.step(action)
            new_data = convert_to_torch_geometric_data(new_observation)
            learn_info = agent.learn((old_data, new_data, action, reward, done))
            old_data = new_data
            environment.render()

            yield dict(
                {
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "episode": episode,
                    "buffer-size": len(agent._replay_buffer),
                },
                **learn_info,
            )


if __name__ == "__main__":
    if input("Would you like to load the most recent model snapshot? [y/n]")[0] == "y":
        agent, model_path = DeepQAgent.load_snapshot()
        print(f"Loaded agent at {model_path}")

    else:
        agent = DeepQAgent()
    # agent.save_snapshot()
    for info in train_agent(agent=agent, episodes=1):
        print(info)
