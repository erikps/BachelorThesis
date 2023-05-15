from pathlib import Path
from copy import deepcopy
from typing import Optional
from uuid import uuid4
import random

import numpy as np
from torch.utils.data import Dataset
import networkx as nx

from src.categoriser import HCategoriser
from src.core import AttackInferenceProblem, WeightedArgumentationFramework


def create_random_graphs(directory: str, number=1000, size=3):
    """ Create random directed graps and save in .apx files. """
    for _ in range(number):
        graph = nx.random_graphs.fast_gnp_random_graph(size, 0.2)
        with open(directory + f"{uuid4()}.apx", "w+") as f:
            for argument in graph.nodes():
                f.write(f"arg({argument}).\n")
            for arg1, arg2 in graph.edges():
                f.write(f"att({arg1},{arg2}).\n")


def random_normal():
    """ randomly assign weights based on normal distribution with results restricted to range [0;1]. """
    random_value = np.random.normal(0.5, 0.1)
    return min(max(random_value, 0), 1)


def random_bimodal():
    """ randomly assign weights based on a kind of "bimodal" distribution based on two normal distributions. """

    random_value = np.random.normal(
        0.25, 0.1) if np.random.random() > 0.5 else np.random.normal(0.75, 0.1)
    return min(max(random_value, 0), 1)


def create_datasets(
        categoriser=HCategoriser(),
        graph_dir: str = "./data/graphs/",
        output_dir: str = "./data/attackinference/",
        randomisers=[random.random, random_normal, random_bimodal]
):
    """Take all graphs from the graph_dir stored in '.apx' format and
    transform them into attack inference problem instances by
    assiging random weights to them and then running the provided
    categoriser. The graphs are then serialised into the output_dir.
    After running this function, the Dataset class below can be used
    to load them again. For each of the frameworks, a random randomiser
    is used to generate the weights.
    """
    source_path = Path(graph_dir)
    output_base_path = Path(output_dir)
    paths = source_path.glob("*.apx")

    for path in paths:
        # find the name of the output file.
        # same name as the input file but without file ending and located in the output directory
        output_filename = path.stem
        output_path = output_base_path.joinpath(output_filename)

        framework = WeightedArgumentationFramework.from_file(
            str(path)
        ).randomise_weights(random.choice(randomisers))

        problem = AttackInferenceProblem(framework, categoriser)
        problem.write_to_disk(str(output_path))


class AttackInferenceDataset(Dataset):
    """Dataset of attack inference problems. The problems first have to be created with create_datasets().
    preload:
        If True, all graphs are immediately loaded into memory. Otherwise, they
        are loaded on a on-demand basis when indexing into the dataset (__getitem__).
    """

    def __init__(self, problems_directory="./data/attackinference/", preload=False):
        self._paths = list(Path(problems_directory).glob("*"))
        self._preloaded_problems = None if not preload else self._preload()

    def _preload(self):
        return [self._load_problem(path) for path in self._paths]

    @staticmethod
    def _load_problem(path: Path):
        return AttackInferenceProblem.read_from_disk(str(path))

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index) -> AttackInferenceProblem:
        if self._preloaded_problems is None:
            return self._load_problem(self._paths[index])

        else:
            return deepcopy(self._preloaded_problems[index])

    @staticmethod
    def example_dataset() -> "AttackInferenceDataset":
        return AttackInferenceDataset("./examples/attackinference")


if __name__ == "__main__":
    # The following code generates problem instances from the graphs in the data folder.
    # Careful: these override the existing problem instances in the "attackinference" folders.

    import os
    for i in range(3, 10):
        try:
            os.mkdir(f"./data/train/train-{i}/attackinference")
        except Exception:
            pass
        create_datasets(
            graph_dir=f"./data/train/train-{i}/graphs", output_dir=f"./data/train/train-{i}/attackinference")

    for i in [10, 25]:
        try:
            os.mkdir(f"./data/test/test-{i}/attackinference")
        except Exception:
            pass
        create_datasets(
            graph_dir=f"./data/test/test-{i}/graphs",
            output_dir=f"./data/test/test-{i}/attackinference",
        )
