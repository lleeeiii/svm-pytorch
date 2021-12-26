import torch
from torch.utils.data import Dataset
import pandas as pd
import os.path

class Loader(Dataset):
    def __init__(self, input, output):
        """
        input = pandas dataframe
        output = pandas dataframe
        After transform to tensor, should have dimensions
            input.shape == [N, input_dim]
            output.shape == [N]
        """
        self.input = torch.tensor(input.values)
        self.input_dim = self.input.shape[1] 
        self.output = torch.tensor(output.values)
    
    def __len__(self):
        return self.output.shape[0]

    def __getitem__(self, index):
        return self.input[index], self.output[index]
