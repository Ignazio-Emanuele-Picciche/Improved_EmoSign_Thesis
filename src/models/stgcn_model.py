import torch
import torch.nn as nn

# Requires torch_geometric for graph convolutions
# from torch_geometric.nn import GCNConv, TemporalConv


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
    """

    def __init__(self, num_class, num_point, num_person=1, in_channels=3):
        super().__init__()
        # Define adjacency matrix A for the skeleton (you need to define a proper one)
        A = torch.eye(num_point)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        # Build multiple blocks with increasing channels
        self.layer1 = STGCNBlock(in_channels, 64, A, residual=False)
        self.layer2 = STGCNBlock(64, 128, A, stride=2)
        self.layer3 = STGCNBlock(128, 256, A, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
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
        return self.fc(x)
