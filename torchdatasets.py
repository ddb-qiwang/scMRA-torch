from __future__ import print_function
import torch.utils.data as data
import torch
from PIL import Image
import numpy as np

class Dataset(data.Dataset):

    def __init__(self, cells, raw, sf, label):
        
        self.cells = cells
        self.raw = raw
        self.sf = sf
        self.labels = label

    def __getitem__(self, index):
        
        cells, raw, sf, label = self.cells[index], self.raw[index], self.sf[index], self.labels[index]

        cells = torch.tensor(cells)
        raw = torch.tensor(raw)
        sf = torch.tensor(sf)
        label = torch.tensor(label)
        
        return cells, raw, sf, label
    
    def __len__(self):
        return len(self.cells)
