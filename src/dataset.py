import glob
from os.path import join

from torch.utils.data import Dataset
from categoriser import HCategoriser

from core import AttackInferenceProblem, WeightedArgumentationFramework


class AttackInferenceDataset(Dataset):
    def __init__ (self, categoriser=HCategoriser(), root="../data/graphs/"):
        self.categoriser = categoriser
        self.paths = glob.glob(join(root, "*.apx"))
        

    def __len__ (self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        framework = WeightedArgumentationFramework.from_file(path)
        return AttackInferenceProblem(framework, self.categoriser)
    


