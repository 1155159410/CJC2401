import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """
    # Params: ~800K

    ReLU; SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    - Epoch 183 | Train Loss: 0.1239, Train Acc: 0.9622 | Val Loss: 0.3346, Val Acc: 0.8676

    LeakyReLU; SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    - Epoch 163 | Train Loss: 0.0796, Train Acc: 0.9732 | Val Loss: 0.3571, Val Acc: 0.8787
    """

    def __init__(self):
        super().__init__()

        # Input size is 33 * 4 = 132, output size is 2 * 2 = 4
        self.fc1 = nn.Linear(132, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 4)

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Flatten the input from (batch_size, 33, 4) to (batch_size, 132)
        x = torch.flatten(x, start_dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)

        return x.view(-1, 2, 2)  # Reshape to (batch_size, 2, 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SharedMLP(nn.Module):
    """
    # Params: ~71M

    LeakyReLU; SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    Epoch 188 | Train Loss: 0.0692, Train Accuracy: 0.9772 | Val Loss: 0.4235, Val Accuracy: 0.9118

    LeakyReLU, Dropout (p=0.5); SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    Epoch 168 | Train Loss: 0.4457, Train Accuracy: 0.7010 | Val Loss: 0.6263, Val Accuracy: 0.2353

    LeakyReLU, Dropout (p=0.3); SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    Epoch 202 | Train Loss: 0.1593, Train Accuracy: 0.9410 | Val Loss: 0.3232, Val Accuracy: 0.8971

    LeakyReLU, Dropout (p=0.2); SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    Epoch 192 | Train Loss: 0.1539, Train Accuracy: 0.9418 | Val Loss: 0.3230, Val Accuracy: 0.8971

    LeakyReLU, Dropout (p=0.2), Batch Normalization; SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    Epoch 165 | Train Loss: 0.1459, Train Accuracy: 0.9426 | Val Loss: 0.3475, Val Accuracy: 0.8787

    LeakyReLU, Batch Normalization; SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    Epoch 169 | Train Loss: 0.0426, Train Accuracy: 0.9874 | Val Loss: 0.4347, Val Accuracy: 0.8934
    """

    def __init__(self, dropout_p: float = 0):
        super().__init__()

        self.dropout_p = dropout_p

        # Shared MLP for each keypoint's (x, y, z, confidence)
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p)
        )

        # After processing all 33 keypoints, aggregate features
        self.fc1 = nn.Linear(512 * 33, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 4)

        self.dropout = nn.Dropout(p=self.dropout_p)

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Apply point-specific MLP to each row (keypoint)
        batch_size, num_points, num_features = x.shape  # (batch_size, 33, 4)

        # Process each keypoint independently using the shared MLP
        x = self.point_mlp(x.view(-1, num_features))  # Shape: (batch_size * 33, 32)

        # Reshape back to (batch_size, 33, 32)
        x = x.view(batch_size, num_points, -1)

        # Flatten the processed keypoints into a single vector for each sample
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers for aggregation
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        output = self.fc4(x)  # Output shape (batch_size, 4)

        return output.view(batch_size, 2, 2)  # Reshape to (batch_size, 2, 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SharedMLPLite(nn.Module):
    """
    # Params: ~600K

    LeakyReLU, Batch Normalization; SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    Epoch 172 | Train Loss: 0.0858, Train Accuracy: 0.9701 | Val Loss: 0.3906, Val Accuracy: 0.8934

    LeakyReLU; SGD (lr=0.01, momentum=0.9, weight_decay=0.0001):
    Epoch 182 | Train Loss: 0.1114, Train Accuracy: 0.9725 | Val Loss: 0.3520, Val Accuracy: 0.8824
    """

    def __init__(self, dropout_p: float = 0, batch_norm: bool = False):
        super().__init__()

        self.dropout_p = dropout_p

        # Shared MLP for each keypoint's (x, y, z, confidence)
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32) if batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p)
        )

        # After processing all 33 keypoints, aggregate features
        self.fc1 = nn.Linear(32 * 33, 512)
        self.bn1 = nn.BatchNorm1d(512) if batch_norm else nn.Identity()
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256) if batch_norm else nn.Identity()
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64) if batch_norm else nn.Identity()
        self.fc4 = nn.Linear(64, 4)

        self.dropout = nn.Dropout(p=self.dropout_p)

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        x[..., -1] = 0

        # Apply point-specific MLP to each row (keypoint)
        batch_size, num_points, num_features = x.shape  # (batch_size, 33, 4)

        # Process each keypoint independently using the shared MLP
        x = self.point_mlp(x.view(-1, num_features))  # Shape: (batch_size * 33, 32)

        # Reshape back to (batch_size, 33, 32)
        x = x.view(batch_size, num_points, -1)

        # Flatten the processed keypoints into a single vector for each sample
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers for aggregation
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        output = self.fc4(x)  # Output shape (batch_size, 4)

        return output.view(batch_size, 2, 2)  # Reshape to (batch_size, 2, 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
