import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
import json
import cv2 

class CombinedDataset(data.Dataset):
    def __init__(
        self,
        datasets_dict
    ):
        self.datasets_dict = datasets_dict
        self.datasets = [self.datasets_dict[key] for key in self.datasets_dict]
        self.dataset_names = [key for key in self.datasets_dict]
        self.lengths = [len(d) for d in self.datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):  
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index], self.dataset_names[i]
        raise IndexError(f'{index} exceeds {self.length}')