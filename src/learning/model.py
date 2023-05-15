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
            message_passing_iterations: int = 5,
            number_hidden_layers: int = 1,
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
            message_input_size, out_channels, number_hidden_layers, out_channels
        )

        # This neural network is used for updating the node representation after each message passing step
        self._update_network = NeuralNetwork(
            self._aggregator_count * out_channels + out_channels,
            out_channels,
            number_hidden_layers,
            out_channels,
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

            # print(node_embeddings)

        return node_embeddings


class ReadoutModel(torch.nn.Module):
    """Responsible for computing node representations and reading out q-values from them.
    parameters:
        channel_count:              number of learned embeddings for each node
        readout_hidden_layer_count: number of hidden layers for the readout network
    """

    def __init__(self, channel_count=64, readout_hidden_layer_count=1) -> None:
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

        # print(data.edge_attr.T[0])

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


def train_pfrl():
    MAX_STEPS = 100000

    dataset = AttackInferenceDataset.example_dataset()
    # dataset = AttackInferenceDataset("./data/train/train-4/attackinference")
    env = AttackInferenceEnvironment(dataset)

    readout_model = ReadoutModel()

    explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)
    # explorer = pfrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.9, end_epsilon=0.1, decay_steps=MAX_STEPS, random_action_func=env.action_space.sample)

    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
    discount_rate = 0.9999

    def phi(instance):
        if isinstance(instance, tuple):
            instance, _ = instance
        return convert_to_torch_geometric_data(instance)

    def create_batch(batch, _, convert):
        return Batch.from_data_list([convert(instance) for instance in batch])

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="")

    optimiser = torch.optim.Adam(readout_model.parameters(), lr=2e-5, eps=1e-2)
    agent = pfrl.agents.DQN(readout_model, optimiser, replay_buffer, discount_rate, explorer,
                            phi=phi, batch_states=create_batch, update_interval=1)

    def on_hook(env, agent: pfrl.agents.DQN, step):
        pass

    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=MAX_STEPS,  # Train the agent for 2000 steps
        eval_n_steps=None,  # We evaluate for episodes, not time
        eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
        train_max_episode_len=500,  # Maximum length of each episode
        eval_interval=1000,  # Evaluate the agent after every 1000 steps
        outdir='result',  # Save everything to 'result' directory
        step_hooks=[on_hook]
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
