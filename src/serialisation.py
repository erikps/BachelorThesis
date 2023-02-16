import random
import re
from typing import List, Generator
import glob
import os

import torch

from categoriser import HCategoriser
from core import WeightedArgumentationFramework, AttackInferenceProblem


def load_problems(dataset_path: str, categoriser=HCategoriser(), seed=None) -> Generator[AttackInferenceProblem, None, None]:
    """ Read all .apx files in the provided folder and return all ArgumentationFrameworks constructed from them. """
    paths = glob.glob(os.path.join(dataset_path, "*.apx"))

    if seed != None:
        random.seed(seed)

    for path in paths:
        framework = WeightedArgumentationFramework.from_file(path)
        yield AttackInferenceProblem(framework, categoriser)


if __name__ == "__main__":
    problems = load_problems("../data/graphs/")
    for node, data in next(problems).framework.graph.nodes(data=True):
        print(data)

        
