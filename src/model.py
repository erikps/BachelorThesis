import torch
from torch_geometric.nn import MessagePassing


class GNN(MessagePassing):
    """ GNN Message passing model. """

    def __init__(self) -> None:
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        # x is of shape [N, in_channels]
        pass

    def propagate(self, edge_index, size=None):
        pass

    def message(self):
        pass
