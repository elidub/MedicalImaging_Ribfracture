import torch
from torch import nn

class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(512*512, 256)
        self.layer_2 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        batch_size, channels, width, height = x.size()
        x = x[:, :, :, 0]
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        X = self.relu(x)
        x = self.layer_2(x)
        return x
    
# Define the CNN model
class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 128 * 128, 128)  # Adjust the input size based on your image size
        self.fc2 = nn.Linear(128, 10)  # Adjust the output size based on your task

    def forward(self, x):
        # print(x.device)



        # # Convolutional layers with ReLU activation and max-pooling
        # x = self.pool(torch.relu(self.conv1(x)))
        # x = self.pool(torch.relu(self.conv2(x)))
        # x = self.pool(torch.relu(self.conv3(x)))
        
        # # Flatten the tensor for fully connected layers
        # x = x.view(-1, 64 * 128 * 128)  # Adjust the size based on your image size
        
        # # Fully connected layers with ReLU activation
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        return x