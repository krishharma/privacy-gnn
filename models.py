"""
Graph neural network models for node classification.
Baselines (LogReg, MLP) are implemented via sklearn in experiment.py.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network for node classification."""

    def __init__(self, ic, h, oc):
        super().__init__()
        self.c1 = GCNConv(ic, h)
        self.c2 = GCNConv(h, oc)

    def forward(self, x, e):
        x = self.c1(x, e)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        return self.c2(x, e)


class SAGE(nn.Module):
    """Two-layer GraphSAGE for node classification."""

    def __init__(self, ic, h, oc):
        super().__init__()
        self.c1 = SAGEConv(ic, h)
        self.c2 = SAGEConv(h, oc)

    def forward(self, x, e):
        x = self.c1(x, e)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        return self.c2(x, e)
