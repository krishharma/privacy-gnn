import torch
import numpy as np
from torch_geometric.data import Data
from models import GCN, SAGE
from epsd_utils import forward_ego_masked, compute_ego_gap

def test_forward_ego_masked():
    print("Running unit tests for forward_ego_masked...")
    # 1. Create a dummy graph
    num_nodes = 10
    num_features = 8
    num_classes = 3
    
    # Fully connected graph for simplicity
    row, col = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    
    # Random features
    torch.manual_seed(42)
    x = torch.randn(num_nodes, num_features)
    y = torch.randint(0, num_classes, (num_nodes,))
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Test cases: GCN and SAGE
    models = [
        ("GCN", GCN(num_features, 16, num_classes)),
        ("GraphSAGE", SAGE(num_features, 16, num_classes))
    ]
    
    for name, model in models:
        model.eval()
        
        with torch.no_grad():
            # Normal forward pass
            logits_normal = model(data.x, data.edge_index)
            p_normal = torch.softmax(logits_normal, dim=-1)
            
            # Masking scenarios
            scenarios = {
                "single_node": [0],
                "subset_nodes": [1, 3, 5, 7],
                "all_nodes": list(range(num_nodes))
            }
            
            for s_name, indices in scenarios.items():
                mask_nodes = torch.zeros(num_nodes, dtype=torch.bool)
                mask_nodes[indices] = True
                
                p_ego = forward_ego_masked(model, data, mask_nodes)
                
                # Check (a): for unmasked nodes, output equals normal pass
                unmasked_indices = (~mask_nodes).nonzero(as_tuple=True)[0]
                if len(unmasked_indices) > 0:
                    diff_unmasked = torch.abs(p_normal[unmasked_indices] - p_ego[unmasked_indices]).max().item()
                    assert diff_unmasked < 1e-5, f"{name} {s_name}: Unmasked nodes differed (max diff: {diff_unmasked})"
                
                # Check (b): for masked nodes, output differs
                masked_indices = mask_nodes.nonzero(as_tuple=True)[0]
                if len(masked_indices) > 0:
                    for idx in masked_indices:
                        if torch.all(data.x[idx] == 0):
                            print(f"Edge case: node {idx} has all zero features, diff might be zero.")
                            continue
                        
                        diff_masked = torch.abs(p_normal[idx] - p_ego[idx]).max().item()
                        assert diff_masked > 1e-5, f"{name} {s_name}: Masked node {idx} did not differ from normal pass!"

    print("All unit tests passed successfully!")

if __name__ == "__main__":
    test_forward_ego_masked()
