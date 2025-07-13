import torch
import torch.nn as nn
import numpy as np


# Perché ST-GCN (Spatial-Temporal Graph Convolutional Network)?

# Modellazione Spaziale: A differenza di un LSTM che tratta i landmark come una lunga
# sequenza di feature, un ST-GCN modella esplicitamente le connessioni fisiche dello scheletro umano
# (es. il gomito è connesso alla spalla e al polso).
# Questo permette al modello di apprendere pattern basati sulla struttura corporea,
# che sono fondamentali per interpretare il linguaggio del corpo e le emozioni.

# Stato dell'Arte: Gli ST-GCN e le loro varianti sono considerati lo stato dell'arte
# per problemi di action recognition e analisi del movimento basati su dati scheletrici.
# Applicarli al riconoscimento delle emozioni è un'estensione potente e moderna.


def get_adjacency_matrix(num_nodes, self_connection=True):
    """
    Generates a normalized adjacency matrix for OpenPose-like skeleton with 25 keypoints.
    """
    # OpenPose BODY_25 keypoint connections
    # fmt: off
    pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
        (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (0, 15),
        (15, 17), (0, 16), (16, 18), (11, 24), (24, 23), (23, 22),
        (14, 21), (21, 20), (20, 19)
    ]
    # fmt: on

    A = np.zeros((num_nodes, num_nodes))
    for i, j in pairs:
        if i < num_nodes and j < num_nodes:
            A[i, j] = 1
            A[j, i] = 1

    if self_connection:
        A += np.eye(num_nodes)

    # Normalize the adjacency matrix
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)
    A_norm = D_mat_inv_sqrt @ A @ D_mat_inv_sqrt

    return torch.from_numpy(A_norm).float()


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal GCN block: apply spatial graph convolution then temporal convolution.
    """

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        # A: adjacency matrix of the skeleton graph (num_nodes x num_nodes)
        self.A = A
        self.gcn = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1)
        )  # placeholder for GCNConv
        self.tcn = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            padding=(4, 0),
            stride=(stride, 1),
        )
        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, channels, time, nodes)
        res = self.residual(x)
        # Spatial graph convolution via adjacency matrix
        # multiply features by adjacency: x[b,c,t,v] * A[v,w] -> x[b,c,t,w]
        x = torch.einsum("bctv,vu->bctu", x, self.A.to(x.device))
        # 1x1 convolution to mix channel features
        x = self.gcn(x)
        # Temporal convolution
        x = self.tcn(x)
        # Batch norm and residual add
        x = self.bn(x + res)
        return self.relu(x)


class STGCN(nn.Module):
    """
    Simplified ST-GCN for skeleton-based emotion recognition.

    The model structure is as follows:
    1. Input Batch Normalization: Normalizes the raw input skeleton data across
       all nodes and channels for each frame.
    2. ST-GCN Block 1: Processes the input data with 64 feature channels.
       - Spatial convolution using the adjacency matrix to learn spatial patterns.
       - Temporal convolution to capture motion patterns.
    3. ST-GCN Block 2: Increases feature channels to 128 and downsamples the
       temporal dimension by a factor of 2 (stride=2), capturing features at a
       larger time scale.
    4. ST-GCN Block 3: Further increases features to 256 and downsamples
       the temporal dimension again, creating a compact and high-level feature
       representation.
    5. Global Average Pooling: Aggregates features across all nodes and time steps
       to produce a single feature vector for each sample in the batch.
    6. Fully Connected Layer: A final linear layer for classification into emotion classes.
    """

    def __init__(
        self, num_class, num_point, num_person=1, in_channels=2, dropout_rate=0.5
    ):
        super().__init__()
        # Define and register adjacency matrix A for the skeleton
        A = get_adjacency_matrix(num_point)
        self.register_buffer("A", A)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        # Build multiple blocks with increasing channels
        self.layer1 = STGCNBlock(in_channels, 64, self.A, residual=False)
        self.layer2 = STGCNBlock(64, 128, self.A, stride=2)
        self.layer3 = STGCNBlock(128, 256, self.A, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # x: (batch, time, nodes, features) -> rearrange to (batch, features, time, nodes)
        x = x.permute(0, 3, 1, 2).contiguous()
        b, c, t, v = x.size()
        # Normalize over channels*nodes for each time-step via BatchNorm1d
        # Reshape to (batch, channels*nodes, time)
        x = x.view(b, c * v, t)
        x = self.data_bn(x)
        # Restore shape (batch, channels, time, nodes)
        x = x.view(b, c, t, v)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(b, -1)
        x = self.dropout(x)
        return self.fc(x)
