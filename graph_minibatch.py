import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import dropout_edge, k_hop_subgraph
from torch_geometric.data import Data
from opacus import PrivacyEngine

class CustomKhopLoader:
    def __init__(self, data, num_hops, input_nodes, batch_size, shuffle=True):
        if input_nodes is None:
            input_nodes = torch.arange(data.num_nodes)
        self.dataset = TensorDataset(input_nodes)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.data = data
        self.num_hops = num_hops
        
    def __iter__(self):
        for batch_nodes in self.loader:
            batch_nodes = batch_nodes[0]
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                batch_nodes, num_hops=self.num_hops, edge_index=self.data.edge_index, relabel_nodes=True
            )
            # Create a Data object containing only the subgraph
            sub_data = Data(x=self.data.x[subset], edge_index=sub_edge_index, y=self.data.y[subset])
            # Store the mapping to target nodes for this batch
            sub_data.mapping = mapping
            sub_data.batch_size = len(batch_nodes)
            sub_data.batch_nodes = batch_nodes
            yield sub_data
            
    def __len__(self):
        return len(self.loader)

def train_gnn_minibatch(
    model,
    train_data,
    device,
    edge_index,
    epochs,
    lr,
    weight_decay,
    batch_size,
    num_neighbors,
    early_stop_patience,
    label_smoothing,
    dropedge_rate,
    val_mask,
    epsd_lambda=0.0,
    epsd_ablation=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = train_data.train_mask.nonzero(as_tuple=False).view(-1)
    
    if val_mask is not None and val_mask.any():
        val_idx = val_mask.nonzero(as_tuple=False).view(-1)
        loader_nodes = torch.cat([train_idx, val_idx])
    else:
        loader_nodes = train_idx
        
    loader = CustomKhopLoader(
        train_data,
        num_hops=len(num_neighbors),
        input_nodes=loader_nodes,
        batch_size=batch_size,
        shuffle=True,
    )
    
    model.train()
    
    metrics = {
        'normal_loss_ep1': float('nan'),
        'epsd_kl_loss_ep1': float('nan'),
        'total_loss_ep1': float('nan'),
        'normal_loss_final': float('nan'),
        'epsd_kl_loss_final': float('nan'),
        'total_loss_final': float('nan'),
    }
    
    for epoch in range(epochs):
        epoch_normal_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total_loss = 0.0
        batches = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            mapping = batch.mapping
            
            is_train = train_data.train_mask[batch.batch_nodes]
            train_mapping = mapping[is_train]
            
            if val_mask is not None and val_mask.any():
                is_val = val_mask[batch.batch_nodes]
                epsd_mapping = mapping[is_val]
            else:
                epsd_mapping = train_mapping
            
            e_idx = batch.edge_index
            if dropedge_rate > 0.0:
                e_idx, _ = dropout_edge(e_idx, p=dropedge_rate, training=model.training)
                
            out = model(batch.x, e_idx)
            
            if train_mapping.numel() > 0:
                if label_smoothing > 0.0:
                    loss = F.cross_entropy(out[train_mapping], batch.y[train_mapping], label_smoothing=label_smoothing)
                else:
                    loss = F.cross_entropy(out[train_mapping], batch.y[train_mapping])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                
            if epsd_lambda > 0.0 and epsd_mapping.numel() > 0:
                if epsd_ablation == "non_ego_consistency":
                    from epsd_utils import forward_random_masked as forward_ablation
                else:
                    from epsd_utils import forward_ego_masked as forward_ablation
                
                p_normal = F.softmax(out, dim=1)
                
                mask_nodes = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
                mask_nodes[epsd_mapping] = True
                p_ego = forward_ablation(model, batch, mask_nodes)
                
                p_v = p_normal[epsd_mapping]
                p_v_ego = p_ego[epsd_mapping]
                
                eps = 1e-8
                p_v = p_v + eps
                p_v_ego = p_v_ego + eps
                p_v = p_v / p_v.sum(dim=-1, keepdim=True)
                p_v_ego = p_v_ego / p_v_ego.sum(dim=-1, keepdim=True)
                
                kl_v_ego = torch.sum(p_v * torch.log(p_v / p_v_ego), dim=-1).mean()
                kl_ego_v = torch.sum(p_v_ego * torch.log(p_v_ego / p_v), dim=-1).mean()
                epsd_loss = 0.5 * kl_v_ego + 0.5 * kl_ego_v
                
                total_loss = loss + epsd_lambda * epsd_loss
                
                epoch_kl_loss += epsd_loss.item()
                epoch_total_loss += total_loss.item()
                loss = total_loss
            else:
                epoch_total_loss += loss.item()
                
            epoch_normal_loss += loss.item() if epsd_lambda <= 0.0 else (loss - epsd_lambda * epsd_loss).item()
            batches += 1
            
            loss.backward()
            optimizer.step()
            
        if epoch == 0:
            metrics['normal_loss_ep1'] = epoch_normal_loss / max(1, batches)
            metrics['epsd_kl_loss_ep1'] = epoch_kl_loss / max(1, batches)
            metrics['total_loss_ep1'] = epoch_total_loss / max(1, batches)
        metrics['normal_loss_final'] = epoch_normal_loss / max(1, batches)
        metrics['epsd_kl_loss_final'] = epoch_kl_loss / max(1, batches)
        metrics['total_loss_final'] = epoch_total_loss / max(1, batches)

    return model, metrics

def infer_logits_minibatch(model, data, edge_index, device, num_neighbors, batch_size, num_nodes, num_classes):
    loader = CustomKhopLoader(
        data,
        num_hops=len(num_neighbors),
        input_nodes=None, 
        batch_size=batch_size,
        shuffle=False
    )
    
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            all_logits.append(out[batch.mapping].cpu())
            
    return torch.cat(all_logits, dim=0)

def train_gnn_dp_minibatch(
    model,
    train_data,
    device,
    edge_index,
    epochs,
    lr,
    weight_decay,
    batch_size,
    num_neighbors,
    max_grad_norm,
    noise_multiplier,
    delta,
    dropedge_rate,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = train_data.train_mask.nonzero(as_tuple=False).view(-1)
    
    loader = CustomKhopLoader(
        train_data,
        num_hops=len(num_neighbors),
        input_nodes=train_idx,
        batch_size=batch_size,
        shuffle=True,
    )
    
    engine = PrivacyEngine()
    model, optimizer, wrapped_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader.loader, # pass the inner PyTorch DataLoader to Opacus!
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
    )
    
    model.train()
    for epoch in range(epochs):
        for batch_nodes in wrapped_loader:
            batch_nodes = batch_nodes[0]
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                batch_nodes, num_hops=len(num_neighbors), edge_index=train_data.edge_index, relabel_nodes=True
            )
            # Create a Data object containing only the subgraph
            sub_data = Data(x=train_data.x[subset], edge_index=sub_edge_index, y=train_data.y[subset])
            sub_data = sub_data.to(device)
            mapping = mapping.to(device)
            
            optimizer.zero_grad()
            
            e_idx = sub_data.edge_index
            if dropedge_rate > 0.0:
                e_idx, _ = dropout_edge(e_idx, p=dropedge_rate, training=model.training)
                
            out = model(sub_data.x, e_idx)
            loss = F.cross_entropy(out[mapping], sub_data.y[mapping])
            loss.backward()
            optimizer.step()
            
    epsilon = engine.get_epsilon(delta)
    if hasattr(model, '_module'):
        return model._module, epsilon
    return model, epsilon
