from pathlib import Path
from copy import deepcopy

from torch.utils.data import Dataset

from src.categoriser import HCategoriser
from src.core import AttackInferenceProblem, WeightedArgumentationFramework


def create_datasets(
        categoriser=HCategoriser(),
        graph_dir: str = "./data/graphs/",
        output_dir: str = "./data/attackinference/",
):
    """Take all graphs from the graph_dir stored in '.apx' format and
    transform them into attack inference problem instances by
    assiging random weights to them and then running the provided
    categoriser. The graphs are then serialised into the output_dir.
    After running this function, the Dataset class below can be used
    to load them again.
    """
    source_path = Path(graph_dir)
    output_base_path = Path(output_dir)
    # paths = glob.glob(join(graph_source_directory, "*.apx"))
    for path in source_path.glob("*.apx"):
        # find the name of the output file.
        # same name as the input file but without file ending and located in the output directory
        output_filename = path.stem
        output_path = output_base_path.joinpath(output_filename)

        framework = WeightedArgumentationFramework.from_file(
            str(path)
        ).randomise_weights()

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
    # create_datasets()
    create_datasets(graph_dir="./examples/", output_dir="./examples/attackinference")

    # import os
    # for i in range(4, 10):
    #     os.mkdir(f"./data/train/train-{i}/attackinference")
    #     create_datasets(graph_dir=f"./data/train/train-{i}/graphs", output_dir=f"./data/train/train-{i}/attackinference")

    # import os

    # for i in [10, 25, 50]:
    #     os.mkdir(f"./data/test/test-{i}/attackinference")
    #     create_datasets(
    #         graph_dir=f"./data/test/test-{i}/graphs",
    #         output_dir=f"./data/test/test-{i}/attackinference",
    #     )
