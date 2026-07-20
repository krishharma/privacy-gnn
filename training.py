"""
GNN training with optional defenses: DropEdge, label smoothing, early stopping, edge sparsification.
"""
import torch
import torch.nn.functional as F

from data import drop_edges, drop_edges_undirected


def train_gnn(model, data, device, epochs=50, lr=0.01, weight_decay=5e-4,
              early_stop_patience=None, label_smoothing=0.0, dropedge_rate=0.0,
              edge_sparsify_rate=0.0, epsd_lambda=0.0, epsd_ablation=None):
    """
    Train the GNN on the given data.
    - early_stop_patience: stop when validation loss does not improve for this many epochs.
    - label_smoothing: smoothing factor for target distribution.
    - dropedge_rate: fraction of edges to drop at each epoch (DropEdge).
    - edge_sparsify_rate: fraction of edges to remove once before training (edge sparsification).
    - epsd_ablation: controls ablations like 'non_ego_consistency' or 'ego_mask_only'.
    """
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    edge_index = data.edge_index
    if edge_sparsify_rate > 0:
        edge_index = drop_edges_undirected(edge_index, rate=edge_sparsify_rate)

    best_loss = float('inf')
    patience_count = 0
    best_state = None

    metrics = {
        'normal_loss_ep1': float('nan'),
        'epsd_kl_loss_ep1': float('nan'),
        'total_loss_ep1': float('nan'),
        'normal_loss_final': float('nan'),
        'epsd_kl_loss_final': float('nan'),
        'total_loss_final': float('nan'),
    }

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if dropedge_rate > 0:
            ei = drop_edges(edge_index, rate=dropedge_rate)
        else:
            ei = edge_index
            
        out = model(data.x, ei)
        
        if epsd_ablation == "ego_mask_only":
            from epsd_utils import forward_ego_masked
            original_edge_index = data.edge_index
            data.edge_index = ei
            try:
                p_ego = forward_ego_masked(model, data, data.train_mask)
            finally:
                data.edge_index = original_edge_index
            # Calculate loss based on ego masked log probabilities
            log_p_ego = torch.log(p_ego[data.train_mask] + 1e-8)
            if label_smoothing > 0.0:
                loss = F.cross_entropy(log_p_ego, data.y[data.train_mask], label_smoothing=label_smoothing)
            else:
                loss = F.nll_loss(log_p_ego, data.y[data.train_mask])
        else:
            if label_smoothing > 0.0:
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], label_smoothing=label_smoothing)
            else:
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

            if epsd_lambda > 0.0:
                if epsd_ablation == "non_ego_consistency":
                    from epsd_utils import forward_random_masked as forward_ablation
                else:
                    from epsd_utils import forward_ego_masked as forward_ablation
                
                p_normal = F.softmax(out, dim=1)
                
                # Use ei to be consistent with any dynamic graph changes, though epsd is run without dropedge
                original_edge_index = data.edge_index
                data.edge_index = ei
                
                # EPSD regularizer should be computed on a proxy non-member set (val_mask) to avoid
                # over-regularizing the target members. If val_mask isn't available, fallback to train_mask.
                epsd_mask = data.val_mask if hasattr(data, 'val_mask') and data.val_mask is not None and data.val_mask.any() else data.train_mask
                
                try:
                    p_ego = forward_ablation(model, data, epsd_mask)
                finally:
                    data.edge_index = original_edge_index
                    
                p_v = p_normal[epsd_mask]
                p_v_ego = p_ego[epsd_mask]
                
                eps = 1e-8
                p_v = p_v + eps
                p_v_ego = p_v_ego + eps
                p_v = p_v / p_v.sum(dim=-1, keepdim=True)
                p_v_ego = p_v_ego / p_v_ego.sum(dim=-1, keepdim=True)
                
                # Symmetric KL (Jensen-Shannon-like) is more stable than one-way KL
                kl_v_ego = torch.sum(p_v * torch.log(p_v / p_v_ego), dim=-1).mean()
                kl_ego_v = torch.sum(p_v_ego * torch.log(p_v_ego / p_v), dim=-1).mean()
                epsd_loss = 0.5 * kl_v_ego + 0.5 * kl_ego_v
                
                total_loss = loss + epsd_lambda * epsd_loss
                
                if epoch == 0:
                    metrics['normal_loss_ep1'] = loss.item()
                    metrics['epsd_kl_loss_ep1'] = epsd_loss.item()
                    metrics['total_loss_ep1'] = total_loss.item()
                metrics['normal_loss_final'] = loss.item()
                metrics['epsd_kl_loss_final'] = epsd_loss.item()
                metrics['total_loss_final'] = total_loss.item()
                
                loss = total_loss
            else:
                if epoch == 0:
                    metrics['normal_loss_ep1'] = loss.item()
                    metrics['epsd_kl_loss_ep1'] = 0.0
                    metrics['total_loss_ep1'] = loss.item()
                metrics['normal_loss_final'] = loss.item()
                metrics['epsd_kl_loss_final'] = 0.0
                metrics['total_loss_final'] = loss.item()

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

    return model, metrics
