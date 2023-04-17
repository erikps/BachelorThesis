import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch_geometric.data import Data
from tqdm import tqdm

from src.dataset import AttackInferenceDataset
from src.model import ReadoutModel, convert_to_torch_geometric_data


class SupervisedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = ReadoutModel(128)

    def forward(self, data: Data):
        embeddings = self.gnn(data)
        readout = self.gnn.readout(
            embeddings, data.edge_index, data.edge_attr
        ).squeeze()
        return readout


def train_supervised():
    dataset = AttackInferenceDataset.example_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervised = SupervisedModel()
    supervised.to(device)

    optimizer = torch.optim.Adam(supervised.parameters(), lr=0.01, weight_decay=5e-4)

    NUMBER_OF_GRAPHS = 1
    NUMBER_OF_EPOCHS = 1000

    losses = []

    supervised.train()
    for _ in tqdm(range(NUMBER_OF_GRAPHS), desc="Problem Instance"):
        # prepare the current problem instance data
        problem = dataset[0]
        data = convert_to_torch_geometric_data(problem)
        data.to(device)

        for epoch in tqdm(
            range(NUMBER_OF_EPOCHS), leave=None, desc="Epoch for Current Problem"
        ):
            optimizer.zero_grad()
            out = supervised(data)
            loss = F.l1_loss(out, data.y)
            losses.append(float(loss))
            loss.backward()
            optimizer.step()
    return supervised

if __name__ == "__main__":
    supervised = train_supervised()
    print(supervised(AttackInferenceDataset.example_dataset()[0].to_torch_geometric()))


