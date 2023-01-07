import os
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from tqdm import tqdm
import tools
import random


class TorchDataset(Dataset):

    def __init__(self, index: int, train: bool = True) -> None:
        super(TorchDataset, self).__init__()
        suffix = "train" if train else "test"
        self.x = np.load(os.path.join(os.getcwd(), "data", "train", str(index), f"x_{suffix}.npy"))
        assert self.x.shape[1] == 200, "error"
        assert self.x.ndim == 2, "error"
        self.y = np.load(os.path.join(os.getcwd(), "data", "train", str(index), f"y_{suffix}.npy"))
        assert self.y.ndim == 1, "error"
        self.x = torch.tensor(self.x).float()
        self.y = torch.tensor(self.y).long()
        print(f"{suffix}_set size = {len(self.x)}")


    def __len__(self) -> int:
        return self.x.size(0)


    def __getitem__(self, index: int) -> Tuple['Tensor', int]:
        return self.x[index], self.y[index]
