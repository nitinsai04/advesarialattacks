import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [batch, 16, 12, 12]
        x = self.pool(F.relu(self.conv2(x)))   # [batch, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
