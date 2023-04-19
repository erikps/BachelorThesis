from collections import deque
from dataclasses import dataclass
from datetime import datetime
from glob import glob
import os
from itertools import count
from random import random
import time
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Optional
from matplotlib.pyplot import logging

import networkx
from networkx.lazy_imports import sys
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
import pfrl
import torch.nn.functional as F
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.tune.logger import pretty_print
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import from_networkx

from src.core import AttackInferenceProblem
from src.dataset import AttackInferenceDataset
from src.envs.environment import AttackInferenceEnvironment, RllibAttackInferenceEnvironment
from src.learning.neuralnetwork import NeuralNetwork


class GraphNeuralNetwork(MessagePassing):
    """
    A message passing model based on the EGNN model from Craandijk and Bex (https://doi.org/10.1609/aaai.v36i5.20497) parameters:
        in_channels:                number of initial node features (weight, desired degree, ...)
        out_channels:               number of node embeddings that are generated for each node after the forward
                                    method is executed
        number_edge_features:       number of features of each edge
        message_passing_iterations: determines how many times the message passing step is repeated.
        number_message_layers:      number of hidden layer in the message creation NN
        message_layer_size          size of a hidden layer in the message creation NN
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            number_edge_features: int,
            message_passing_iterations: int = 1,
            number_message_layers: int = 1,
            message_layer_size=100,
    ):
        aggregators = ["std", "min", "max", "mean"]
        # aggregators = "add"

        super().__init__(aggr=aggregators)

        self._aggregator_count = len(aggregators)

        self._message_passing_iterations = message_passing_iterations

        message_input_size = (
                number_edge_features + out_channels * 2
        )  # Size of the raw inputs to the message encoding network

        # The initial layer performs an initial transformation on the node features before any message passing takes place
        self._initial_linear = torch.nn.Linear(in_channels, out_channels)

        # The purpose of the message network is to learn how to encode messages given both node embeddings and the edge features of the relevant edge
        self._message_network = NeuralNetwork(
            message_input_size, out_channels, number_message_layers, message_layer_size
        )

        # This neural network is used for updating the node representation after each message passing step
        self._update_network = NeuralNetwork(
            self._aggregator_count * out_channels + out_channels,
            out_channels,
            number_message_layers,
            message_layer_size,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self._initial_linear.reset_parameters()
        self._message_network.reset_parameters()

    def message(
            self,
            node_embeddings_i: torch.Tensor,
            node_embeddings_j: torch.Tensor,
            edge_features: torch.Tensor,
    ):
        """Create the message for a given node/node.
            node_embeddings_i: the node that is going to receive the message
            node_embeddings_j: the neighbounring node, sending out the message
            edge_features:     the features of the attack/edge

        note: node_embeddings_{i,j} are passed "magically" from the call to self.propagate
              in the self.forward method, specifically from the x parameter
        """

        # Consolidate node embeddings and the edge feature into one tensor
        message_input = torch.cat(
            (node_embeddings_i, node_embeddings_j, edge_features), dim=1
        )

        # Use the message encoding network to generate the message
        return self._message_network(message_input)

    def update(
            self,
            inputs: torch.Tensor,
            node_embeddings: torch.Tensor
    ):
        """Update each node based on the aggregated messages using a dense NN."""
        return self._update_network(torch.cat((inputs, node_embeddings), dim=1))

    def forward(
            self, node_features: torch.Tensor, edge_features: torch.Tensor, edge_index
    ):
        """Main method called to start propagation. Returns the final node embeddings."""

        # apply initial linear transformation of node features to get the first node embeddings
        # this expands the size of each node's vector
        node_embeddings = self._initial_linear(node_features)

        # do multiple iterations of the massage passing step
        for _ in range(self._message_passing_iterations):
            # all arguments passed into propagate are passed along into 'update' and 'message'

            node_embeddings = F.relu(node_embeddings)
            node_embeddings = self.propagate(
                node_embeddings=node_embeddings,
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
            )

        return node_embeddings


class ReadoutModel(torch.nn.Module):
    """Responsible for computing node representations and reading out q-values from them.
    parameters:
        channel_count:              number of learned embeddings for each node
        readout_hidden_layer_count: number of hidden layers for the readout network
    """

    def __init__(self, channel_count=128, readout_hidden_layer_count=1) -> None:
        super().__init__()
        number_of_node_features = 3  # weight, ground_truth_degree and predicted_degree
        number_of_edge_features = 2  # predicted_edge, has_flipped

        # The GNN layers perform message passing to create node embeddings for each node
        self.gnn_layer = GraphNeuralNetwork(
            number_of_node_features, channel_count, number_of_edge_features
        )

        # The readout network takes a (attacker, attacked) pair and returns a q-value for it
        self.readout_network = NeuralNetwork(
            input_layer_size=2 * channel_count + number_of_edge_features,
            output_layer_size=1,
            hidden_layer_size=channel_count,
            hidden_layer_count=readout_hidden_layer_count,
        )

    def forward(self, data: Data):
        """Use the readout network to determine the edge outputs
        parameters:
            node_embeddings: These are the node embeddings (channels) computed by the GNN of shape [node_count, channel_count]
            edge_index:      Tensor containing [0] the attackers and [1] the attacked for each attack in the graph
            edge_features:   The features associated with attacks, for example if it is supposed to exist
        """
        
        # first, need to calculate embeddings using the gnn
        node_features = data.x
        edge_index = data.edge_index
        edge_features = data.edge_attr
        node_embeddings = self.gnn_layer(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )

        # first, separate the attacker and attacked nodes from the edge_index.
        attackers, attacked = (
            node_embeddings[edge_index[0]],
            node_embeddings[edge_index[1]],
        )

        # readout_input is of shape [edge_count, channel_count * 2 + edge_feature_count]
        readout_input = torch.cat((attackers, attacked, edge_features), dim=1)

        result = self.readout_network(readout_input)
        result = pfrl.action_value.DiscreteActionValue(result)
        return result


class RllibReadoutModel(ReadoutModel, TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        ReadoutModel.__init__(self)
        self._output = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        print("FORWARD")
        print("INPUT DICT", input_dict["obs"])

    def value_function(self, *args, **kwargs):
        print("VALUE FUNCTION")
        print(args, kwargs)


def convert_to_torch_geometric_data(problem: AttackInferenceProblem) -> Data:
    """ Convert an attack inference problem into the pytorch-geometric
    representation needed for learning."""

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


class TrainingLogger:
    def __init__(self, agent_name: str):
        self._file_path = f"./log/train/{agent_name}.csv"

        already_exists = os.path.exists(self._file_path)
        self._file = open(self._file_path, "a")
        if not already_exists:
            self._file.write("step_count,done,runtime_seconds,timeout,epsilon\n")
            self._file.flush()  # make the changes available immediatly

    def write(self, step_count: int, done: bool, runtime_seconds: float, timeout: bool, epsilon: float):
        self._file.write(f"{step_count},{done},{runtime_seconds},{timeout},{epsilon}\n")
        self._file.flush()

    def __del__(self):
        self._file.close()


class DeepQAgent:
    pass


#
# def train_agent(
#         agent_name: str,
#         agent: DeepQAgent = DeepQAgent(),
#         environment: AttackInferenceEnvironment = AttackInferenceEnvironment(
#             AttackInferenceDataset.example_dataset(), render_mode="human"
#         ),
#         episodes: int = 10,
#         timeout_seconds: Optional[int] = 15 * 60,  # 15 minutes default timeout
#         autosave_interval: Optional[int] = None,
# ):
#     """Train the agent using the provided environment."""
#
#     def prefill_experience_buffer():
#         # fill the experience buffer with random experiences
#         old_observation, _ = environment.reset(new_instance=False)
#         old_data = convert_to_torch_geometric_data(old_observation)
#
#         for _ in range(agent._replay_buffer.maxlen):
#             action = environment.action_space.sample()
#             new_observation, reward, done, _, _ = environment.step(action)
#             new_data = convert_to_torch_geometric_data(new_observation)
#             agent.add_experience((old_data, new_data, action, reward, done))
#             old_data = new_data
#
#         environment.reset(new_instance=False)
#
#     prefill_experience_buffer()
#
#     logger = TrainingLogger(agent_name)
#
#     for episode in range(episodes):
#         episode_start_time = time.time()
#         time_delta = 0
#
#         agent._total_training_episodes += 1
#         old_observation, _ = environment.reset()
#         old_data = convert_to_torch_geometric_data(old_observation)
#
#         steps = 0
#
#         # check if an autosave is due
#         if (
#                 episode != 0
#                 and autosave_interval is not None
#                 and episode % autosave_interval == 0
#         ):
#             agent.save_snapshot(agent_name, postfix=f"autosave{episode}")
#
#         done = False
#         timed_out = False
#
#         while not done and not timed_out:
#             action = agent.get_action(old_data)
#             new_observation, reward, done, _, _ = environment.step(action)
#             new_data = convert_to_torch_geometric_data(new_observation)
#             learn_info = agent.learn((old_data, new_data, action, reward, done))
#             old_data = new_data
#             environment.render()
#
#             if steps % 10 == 0:
#                 print(
#                     f"loss {learn_info['loss']:.5f} (episode: {episode}, step_count: {steps}, epsilon: {agent._epsilon})")
#             steps += 1
#
#             time_delta = time.time() - episode_start_time
#             if (
#                     timeout_seconds is not None
#                     and timeout_seconds < time_delta
#             ):
#                 timed_out = True
#
#         logger.write(steps, done, time_delta, timed_out, agent._epsilon)
#
#         # the agent updates the epsilon after an episode is done
#         agent.notify_episode_done()


@dataclass
class EvaluationResult:
    successful_samples: int  # number of successful samples
    total_samples: int  # total number of evaluation samples


def evaluate_agent(
        agent: DeepQAgent,
        environment: AttackInferenceEnvironment,
        number_of_samples: int,
        timeout_seconds: Optional[float] = None,
        maximum_steps: Optional[int] = None,
):
    successes = 0

    def is_unsuccessful(steps: int, time: float, start_time: float):
        # check if the attempt was unsuccessful
        steps_condition = maximum_steps is not None and steps >= maximum_steps
        time_condition = (
                timeout_seconds is not None and timeout_seconds >= time - start_time
        )
        return steps_condition or time_condition

    for current_sample in range(number_of_samples):
        print("Current sample", current_sample)
        steps = 0
        start_time = time.time()
        current_time = start_time
        done = False

        old_observation, _ = environment.reset(new_instance=False)
        old_data = convert_to_torch_geometric_data(old_observation)

        while not is_unsuccessful(steps, current_time, start_time) and not done:
            if timeout_seconds is not None:
                current_time = time.time()

            # get a greedy action
            action = agent.get_action(old_data, do_exploration=False)

            print(action)
            new_observation, _, done, _, _ = environment.step(action)
            new_data = convert_to_torch_geometric_data(new_observation)

            old_data = new_data

            steps += 1

            if done:
                successes += 1

    return EvaluationResult(
        successful_samples=successes, total_samples=number_of_samples
    )


def visualise_q_values(agent: DeepQAgent, problem: AttackInferenceProblem):
    def remove_fake_edges(graph: networkx.DiGraph) -> networkx.DiGraph:
        # remove all edges that are neither actual or predicted
        edges = list(graph.edges.data())
        for a, b, data in edges:
            actual_edge = data["actual_edge"]
            predicted_edge = data["predicted_edge"]
            if not (predicted_edge):
                graph.remove_edge(a, b)
            else:
                predicted_edge = random.random() > 0.3
                color = 1 if predicted_edge else 0
                graph.add_edge(a, b, color=color, predicted_edge=predicted_edge)
        return graph

    data = problem.to_torch_geometric()
    q_values = agent._online_model(data)

    print(list(agent._online_model.readout_network.parameters()))
    print(q_values)


def train_pfrl():
    MAX_STEPS = 10000

    env = AttackInferenceEnvironment(AttackInferenceDataset.example_dataset())

    readout_model = ReadoutModel()

    # explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)
    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=0.9, end_epsilon=0.1, decay_steps=MAX_STEPS, random_action_func=env.action_space.sample)

    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
    discount_rate = 0.99

    def phi(instance):
        if isinstance(instance, tuple):
            instance, _ = instance
        return convert_to_torch_geometric_data(instance)

    def create_batch(batch, _, convert):
        return Batch.from_data_list([convert(instance) for instance in batch])

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format="")

    optimiser = torch.optim.Adam(readout_model.parameters(), eps=1e-2)
    agent = pfrl.agents.DoubleDQN(readout_model, optimiser, replay_buffer, discount_rate, explorer,
                            phi=phi, batch_states=create_batch)

   
    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=MAX_STEPS,           # Train the agent for 2000 steps
        eval_n_steps=None,       # We evaluate for episodes, not time
        eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
        train_max_episode_len=200,  # Maximum length of each episode
        eval_interval=1000,   # Evaluate the agent after every 1000 steps
        outdir='result',      # Save everything to 'result' directory
    ) 

def train():
    total_steps = 10

    ModelCatalog.register_custom_model("readout_model", RllibReadoutModel)
    ray.init()

    algorithm = (
        PPOConfig()
        .environment(RllibAttackInferenceEnvironment, env_config={"dataset": AttackInferenceDataset.example_dataset()})
        .framework("torch")
        .training(model={"custom_model": "readout_model"})
        .build()
    )

    for step in range(total_steps):
        result = algorithm.train()
        print(pretty_print(result))


if __name__ == "__main__":
    train_pfrl()
