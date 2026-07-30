"""
Graph neural network models for node classification.
Baselines (LogReg, MLP) are implemented via sklearn in experiment.py.

GatedGCN / GatedSAGE implement HCAG (Homophily-Conditioned Aggregation Gates)
for the SAMI defense: soft per-edge gates computed from endpoint embeddings and
per-node structural risk down-weight aggregation paths likely to create
train/test posterior asymmetry.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


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


class GAT(nn.Module):
    """Two-layer Graph Attention Network for node classification."""

    def __init__(self, ic, h, oc, heads: int = 4):
        super().__init__()
        self.c1 = GATConv(ic, h // heads, heads=heads, dropout=0.5)
        self.c2 = GATConv(h, oc, heads=1, concat=False, dropout=0.5)

    def forward(self, x, e):
        x = self.c1(x, e)
        x = F.elu(x)
        x = F.dropout(x, 0.5, training=self.training)
        return self.c2(x, e)


class EdgeGate(nn.Module):
    """
    HCAG gate: g_uv = sigmoid(MLP([h_u || h_v || r_u || r_v])) in (0, 1).
    Initialized so gates start near 1 (identity behavior) for stable training.
    """

    def __init__(self, h):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * h + 2, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        # Bias the final layer so sigmoid(out) starts near ~0.88 (open gates).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 2.0)

    def forward(self, hidden, edge_index, risk):
        src, dst = edge_index[0], edge_index[1]
        feats = torch.cat(
            [
                hidden[src],
                hidden[dst],
                risk[src].unsqueeze(1),
                risk[dst].unsqueeze(1),
            ],
            dim=1,
        )
        return torch.sigmoid(self.mlp(feats)).squeeze(-1)


class GatedGCN(nn.Module):
    """
    Two-layer GCN with an HCAG gate on the second aggregation layer.
    Falls back to plain GCN behavior when risk is None.
    """

    supports_risk = True

    def __init__(self, ic, h, oc):
        super().__init__()
        self.c1 = GCNConv(ic, h)
        self.c2 = GCNConv(h, oc)
        self.gate = EdgeGate(h)

    def forward(self, x, e, risk=None):
        x = self.c1(x, e)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        if risk is None:
            return self.c2(x, e)
        g = self.gate(x, e, risk.to(x.device))
        return self.c2(x, e, edge_weight=g)


class GatedSAGE(nn.Module):
    """
    Two-layer GraphSAGE where the second layer uses a gated weighted-mean
    aggregation: agg_v = sum_u g_uv h_u / sum_u g_uv; out = W_self h_v + W_nbr agg_v.
    Falls back to plain SAGE behavior when risk is None.
    """

    supports_risk = True

    def __init__(self, ic, h, oc):
        super().__init__()
        self.c1 = SAGEConv(ic, h)
        self.c2 = SAGEConv(h, oc)  # used only in the risk=None fallback path
        self.gate = EdgeGate(h)
        self.lin_self = nn.Linear(h, oc)
        self.lin_nbr = nn.Linear(h, oc)

    def forward(self, x, e, risk=None):
        x = self.c1(x, e)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        if risk is None:
            return self.c2(x, e)
        g = self.gate(x, e, risk.to(x.device))
        src, dst = e[0], e[1]
        n = x.size(0)
        num = torch.zeros(n, x.size(1), device=x.device)
        num.index_add_(0, dst, x[src] * g.unsqueeze(1))
        den = torch.zeros(n, device=x.device)
        den.index_add_(0, dst, g)
        agg = num / den.clamp(min=1e-8).unsqueeze(1)
        return self.lin_self(x) + self.lin_nbr(agg)
