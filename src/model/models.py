import torch
from torch import nn

class DummyNetwork(nn.Module):
    # This is a dummy network, such that the code runs
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(512*512, 256)
        self.layer_2 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        batch_size, channels, width, height = x.size()
        x = x[:, :, :, 0] # Only select the first slice
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        X = self.relu(x)
        x = self.layer_2(x)
        return x
    