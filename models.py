from operator import mul

import torch
from torch import nn
from torch.nn import functional as F

OUT_SHAPE = (4, 2)  # (No. of posture classes, 2)


class MLP(nn.Module):
    """
    # Params: 4664
    """

    def __init__(self):
        super().__init__()

        # Input shape is (batch_size, 33, 4)
        self.fc1 = nn.Linear(33 * 4, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, mul(*OUT_SHAPE))

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Flatten the input from (batch_size, 33, 4) to (batch_size, 33 * 4)
        x = torch.flatten(x, start_dim=1)

        # Pass through fully connected layers with F.leaky_relu
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        # Final fully connected layer to reduce to the desired output size
        x = self.fc4(x)

        # Reshape the output to (batch_size, *OUT_SHAPE)
        x = x.view(x.size(0), *OUT_SHAPE)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SharedMLPLite(nn.Module):
    """
    # Params: 4560
    """

    def __init__(self):
        super().__init__()

        # Conv1D layers to go from input (B, 4, 33) to (B, 4, 33)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(4 * 33, 32)
        self.fc2 = nn.Linear(32, mul(*OUT_SHAPE))

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Transpose the input from (B, 33, 4) to (B, 4, 33)
        x = x.transpose(1, 2)

        # Apply Conv1D layers with F.leaky_relu
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        # Flatten the output from (B, 4, 33) to (B, 4 * 33)
        x = torch.flatten(x, start_dim=1)

        # Pass through fully connected layers with F.leaky_relu
        x = F.leaky_relu(self.fc1(x))

        # Final fully connected layer to reduce to the desired output size
        x = self.fc2(x)

        # Reshape the output to (B, *OUT_SHAPE)
        x = x.view(x.size(0), *OUT_SHAPE)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SharedMLP_(nn.Module):
    """
    # Params: 169704

    An implementation of `SharedMLP` without using `Conv1D`.
    """

    def __init__(self, dropout_p: float = 0.):
        super().__init__()

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        # Shared MLP for each keypoint's (x, y, z, confidence)
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            self.dropout
        )

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 33, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, mul(*OUT_SHAPE))

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Mask the `confidence` column
        x[..., 3] = 0

        # Unpack the input tensor dimensions
        batch_size, num_points, num_features = x.shape

        # Process each keypoint independently using the shared MLP
        x = self.point_mlp(x.view(-1, num_features))  # (batch_size * 33, 4) -> (batch_size * 33, out_features)

        # Reshape back to (batch_size, 33, out_features)
        x = x.view(batch_size, num_points, -1)

        # Flatten the processed keypoints into a single vector for each sample
        x = torch.flatten(x, start_dim=1)

        # Pass through fully connected layers with F.leaky_relu and Dropout
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        # Final fully connected layer to reduce to the desired output size
        x = self.fc3(x)

        # Reshape the output to (B, *OUT_SHAPE)
        x = x.view(batch_size, *OUT_SHAPE)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SharedMLP(nn.Module):
    """
    # Params: 169704
    """

    def __init__(self, dropout_p: float = 0.):
        super().__init__()

        # Conv1D layers to go from input (B, 4, 33) to (B, 256, 33)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 33, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, mul(*OUT_SHAPE))

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Mask the `confidence` column
        x[..., 3] = 0

        # Transpose the input from (B, 33, 4) to (B, 4, 33)
        x = x.transpose(1, 2)

        # Apply Conv1D layers with F.leaky_relu
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        # Apply dropout after Conv layers
        x = self.dropout(x)

        # Flatten the output from (B, 256, 33) to (B, 256 * 33)
        x = torch.flatten(x, start_dim=1)

        # Pass through fully connected layers with F.leaky_relu and Dropout
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        # Final fully connected layer to reduce to the desired output size
        x = self.fc3(x)

        # Reshape the output to (B, *OUT_SHAPE)
        x = x.view(x.size(0), *OUT_SHAPE)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
