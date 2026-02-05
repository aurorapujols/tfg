import torch
import torch.nn as nn
import torch.nn.functional as F

class SSLBackbone(nn.Module):
    def __init__(self, out_dim, scale_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, scale_dim, 3, padding=1),
            nn.BatchNorm2d(scale_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(scale_dim, scale_dim*2, 3, padding=1),
            nn.BatchNorm2d(scale_dim*2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(scale_dim*2, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)

        # Global average pooling → fixed (batch, out_dim)
        x = x.mean(dim=[2, 3])
        return x

class SSLProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, projection_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, projection_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, dim=1)