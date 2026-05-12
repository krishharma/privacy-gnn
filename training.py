"""
GNN training with optional defenses: DropEdge, label smoothing, early stopping, edge sparsification.
"""
import torch
import torch.nn.functional as F

from data import drop_edges, drop_edges_undirected


def train_gnn(model, data, device, epochs=50, lr=0.01, weight_decay=5e-4,
              early_stop_patience=None, label_smoothing=0.0, dropedge_rate=0.0,
              edge_sparsify_rate=0.0):
    """
    Train the GNN on the given data.
    - early_stop_patience: stop when validation loss does not improve for this many epochs.
    - label_smoothing: smoothing factor for target distribution.
    - dropedge_rate: fraction of edges to drop at each epoch (DropEdge).
    - edge_sparsify_rate: fraction of edges to remove once before training (edge sparsification).
    """
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    edge_index = data.edge_index
    if edge_sparsify_rate > 0:
        edge_index = drop_edges_undirected(edge_index, edge_sparsify_rate)

    num_classes = data.y.max().item() + 1
    best_loss = 1e9
    patience_count = 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Apply DropEdge during training if requested
        ei = drop_edges(edge_index, dropedge_rate) if dropedge_rate > 0 else edge_index
        out = model(data.x, ei)

        if label_smoothing > 0:
            log_p = F.log_softmax(out[data.train_mask], 1)
            smooth = torch.full_like(log_p, label_smoothing / num_classes)
            smooth.scatter_(1, data.y[data.train_mask].unsqueeze(1), 1 - label_smoothing + label_smoothing / num_classes)
            loss = -(smooth * log_p).sum(1).mean()
        else:
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        if early_stop_patience is not None:
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
            if patience_count >= early_stop_patience and best_state is not None:
                model.load_state_dict(best_state)
                break

    return model
