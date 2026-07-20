import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from models import GCN, SAGE
import numpy as np

def forward_ego_masked(model, data, mask_nodes):
    """
    Computes the forward pass where the first-layer self-feature contribution
    is zeroed out for the nodes in mask_nodes.
    
    This is implemented such that:
    1. Unmasked nodes' predictions match the normal forward pass EXACTLY.
    2. Masked nodes' predictions match as if ONLY that specific node was masked.
    3. It runs in a single batched pass for computational efficiency.
    
    Args:
        model: GCN or SAGE model
        data: PyG Data object
        mask_nodes: boolean tensor of shape (N,) indicating which nodes to ego-mask
    
    Returns:
        p_ego: Softmax probability vector of shape (N, C)
    """
    is_gcn = isinstance(model, GCN)
    is_sage = isinstance(model, SAGE)
    
    if not (is_gcn or is_sage):
        raise ValueError("Model must be GCN or SAGE")

    device = data.x.device
    N = data.num_nodes

    # 1. Compute normal first layer representation
    if is_gcn:
        x1_normal = model.c1(data.x, data.edge_index)
    else:
        x1_normal = model.c1(data.x, data.edge_index)
        
    x1_normal = F.relu(x1_normal)
    x1_normal_dropped = F.dropout(x1_normal, p=0.5, training=model.training)

    # 2. Compute ego-masked first layer representation
    if is_gcn:
        # GCN: Intercept the normalization and zero out self-loops for masked nodes
        edge_index_norm, edge_weight_norm = gcn_norm(
            data.edge_index, None, N, improved=False, add_self_loops=True, dtype=data.x.dtype
        )
        
        self_loop_mask = (edge_index_norm[0] == edge_index_norm[1])
        nodes_are_masked = mask_nodes[edge_index_norm[0]]
        target_edges = self_loop_mask & nodes_are_masked
        
        edge_weight_ego = edge_weight_norm.clone()
        edge_weight_ego[target_edges] = 0.0
        
        old_normalize = model.c1.normalize
        model.c1.normalize = False
        try:
            x1_ego = model.c1(data.x, edge_index_norm, edge_weight_ego)
        finally:
            model.c1.normalize = old_normalize
            
    elif is_sage:
        # GraphSAGE: Pass a tuple (x_src, x_dst) to the first layer, where x_dst is zeroed for masked nodes
        x_src = data.x
        x_dst = data.x.clone()
        x_dst[mask_nodes] = 0.0
        x1_ego = model.c1((x_src, x_dst), data.edge_index)

    x1_ego = F.relu(x1_ego)
    x1_ego_dropped = F.dropout(x1_ego, p=0.5, training=model.training)

    # 3. Create a mixed representation for the second layer
    # We want neighbors to see the normal representation, but the node itself to see its ego representation.
    
    if is_gcn:
        # Normal logits
        logits_normal = model.c2(x1_normal_dropped, data.edge_index)
        
        # We need to replace the self-loop contribution.
        # logits_v = sum_{u!=v} A_{vu} (x1_normal_u W2) + A_{vv} (x1_ego_v W2)
        # Therefore: logits_ego = logits_normal - A_{vv} (x1_normal_v W2) + A_{vv} (x1_ego_v W2)
        
        # Get A_{vv}
        edge_index_norm_c2, edge_weight_norm_c2 = gcn_norm(
            data.edge_index, None, N, improved=False, add_self_loops=True, dtype=x1_normal_dropped.dtype
        )
        self_loop_mask_c2 = (edge_index_norm_c2[0] == edge_index_norm_c2[1])
        # Sort by node index just to be safe
        self_loop_edges = edge_index_norm_c2[:, self_loop_mask_c2]
        self_loop_weights = edge_weight_norm_c2[self_loop_mask_c2]
        
        # A_vv for each node
        A_vv = torch.zeros(N, dtype=x1_normal_dropped.dtype, device=device)
        A_vv[self_loop_edges[0]] = self_loop_weights
        
        # Project representations
        x1_normal_proj = model.c2.lin(x1_normal_dropped)
        x1_ego_proj = model.c2.lin(x1_ego_dropped)
        
        # Compute delta
        delta = A_vv.unsqueeze(-1) * (x1_ego_proj - x1_normal_proj)
        
        # Apply mask so we only perturb mask_nodes (though delta should be 0 for unmasked anyway)
        delta = delta * mask_nodes.unsqueeze(-1).to(delta.dtype)
        
        logits = logits_normal + delta

    elif is_sage:
        # GraphSAGE accepts tuples natively.
        # x_src (messages) is normal, x_dst (root update) uses ego for masked nodes.
        x_dst = x1_normal_dropped.clone()
        x_dst[mask_nodes] = x1_ego_dropped[mask_nodes]
        
        logits = model.c2((x1_normal_dropped, x_dst), data.edge_index)
        
    return F.softmax(logits, dim=-1)

def compute_ego_gap(model, data, node_indices):
    """
    Computes the ego-gap for a set of nodes.
    
    Args:
        model: GCN or SAGE model
        data: PyG Data object
        node_indices: tensor of indices to compute the gap for.
        
    Returns:
        g_v: array of ego-gaps for the specified nodes
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, GCN):
            logits = model(data.x, data.edge_index)
        elif isinstance(model, SAGE):
            logits = model(data.x, data.edge_index)
        else:
            raise ValueError("Unsupported model")
            
        p_normal = F.softmax(logits, dim=-1)
        
        # Batch optimization: mask all target nodes simultaneously
        mask_nodes = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
        mask_nodes[node_indices] = True
        
        p_ego = forward_ego_masked(model, data, mask_nodes)
        
        # Extract the distributions for the target nodes
        p_v = p_normal[node_indices]
        p_v_ego = p_ego[node_indices]
        
        # Compute KL divergence
        eps = 1e-8
        p_v = p_v + eps
        p_v_ego = p_v_ego + eps
        
        # Normalize to ensure they sum to 1 after adding eps
        p_v = p_v / p_v.sum(dim=-1, keepdim=True)
        p_v_ego = p_v_ego / p_v_ego.sum(dim=-1, keepdim=True)
        
        g_v = torch.sum(p_v * torch.log(p_v / p_v_ego), dim=-1).cpu().numpy()
        
    return g_v

def forward_random_masked(model, data, mask_nodes):
    """
    Computes the forward pass where one random neighbor's contribution
    is zeroed out for the nodes in mask_nodes (Ablation Control).
    """
    is_gcn = isinstance(model, GCN)
    is_sage = isinstance(model, SAGE)
    
    if not (is_gcn or is_sage):
        raise ValueError("Model must be GCN or SAGE")

    device = data.x.device
    N = data.num_nodes

    if is_gcn:
        x1_normal = model.c1(data.x, data.edge_index)
    else:
        x1_normal = model.c1(data.x, data.edge_index)
        
    x1_normal = F.relu(x1_normal)
    x1_normal_dropped = F.dropout(x1_normal, p=0.5, training=model.training)

    if is_gcn:
        edge_index_norm, edge_weight_norm = gcn_norm(
            data.edge_index, None, N, improved=False, add_self_loops=True, dtype=data.x.dtype
        )
        
        # Pick one random edge per node
        nodes_to_mask = torch.nonzero(mask_nodes).squeeze(-1)
        target_edges = torch.zeros_like(edge_weight_norm, dtype=torch.bool)
        
        # for each node, find its incoming edges and mask a random one (that isn't self-loop if possible)
        # For efficiency, we just randomize a mask
        for v in nodes_to_mask:
            edges = torch.nonzero(edge_index_norm[1] == v).squeeze(-1)
            if len(edges) > 0:
                target_edges[edges[torch.randint(0, len(edges), (1,)).item()]] = True
                
        edge_weight_ego = edge_weight_norm.clone()
        edge_weight_ego[target_edges] = 0.0
        
        old_normalize = model.c1.normalize
        model.c1.normalize = False
        try:
            x1_ego = model.c1(data.x, edge_index_norm, edge_weight_ego)
        finally:
            model.c1.normalize = old_normalize
            
    elif is_sage:
        # Just mask a random neighbor's feature before aggregation for simplicity
        x_src = data.x
        x_dst = data.x.clone()
        # To strictly drop a random edge, we would need to modify edge_index,
        # but for SAGE simply injecting random feature noise to target nodes is an alternative, 
        # or we just remove a random edge from edge_index for the first layer
        ei = data.edge_index.clone()
        target_edges = []
        nodes_to_mask = torch.nonzero(mask_nodes).squeeze(-1)
        for v in nodes_to_mask:
            edges = torch.nonzero(ei[1] == v).squeeze(-1)
            if len(edges) > 0:
                target_edges.append(edges[torch.randint(0, len(edges), (1,)).item()])
        if target_edges:
            mask = torch.ones(ei.size(1), dtype=torch.bool, device=device)
            mask[torch.tensor(target_edges, device=device)] = False
            ei = ei[:, mask]
            
        x1_ego = model.c1(data.x, ei)

    x1_ego = F.relu(x1_ego)
    x1_ego_dropped = F.dropout(x1_ego, p=0.5, training=model.training)

    if is_gcn:
        logits_normal = model.c2(x1_normal_dropped, data.edge_index)
        
        edge_index_norm_c2, edge_weight_norm_c2 = gcn_norm(
            data.edge_index, None, N, improved=False, add_self_loops=True, dtype=x1_normal_dropped.dtype
        )
        # Using self loop logic identically for second layer replacement so the only difference is layer 1
        self_loop_mask_c2 = (edge_index_norm_c2[0] == edge_index_norm_c2[1])
        self_loop_edges = edge_index_norm_c2[:, self_loop_mask_c2]
        self_loop_weights = edge_weight_norm_c2[self_loop_mask_c2]
        
        A_vv = torch.zeros(N, dtype=x1_normal_dropped.dtype, device=device)
        A_vv[self_loop_edges[0]] = self_loop_weights
        
        x1_normal_proj = model.c2.lin(x1_normal_dropped)
        x1_ego_proj = model.c2.lin(x1_ego_dropped)
        
        A_vv_expanded = A_vv.unsqueeze(1)
        logits = logits_normal - A_vv_expanded * x1_normal_proj + A_vv_expanded * x1_ego_proj
        
    elif is_sage:
        logits_normal = model.c2(x1_normal_dropped, data.edge_index)
        x_dst = x1_ego_dropped.clone()
        x_dst[~mask_nodes] = x1_normal_dropped[~mask_nodes]
        logits = model.c2((x1_normal_dropped, x_dst), data.edge_index)
        
    return F.softmax(logits, dim=-1)
