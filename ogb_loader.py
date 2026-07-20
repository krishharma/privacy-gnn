import os
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

# Patch torch.load to bypass weights_only=True issue in PyTorch 2.6+
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

MINIBATCH_DATASETS = frozenset(["ogbn-arxiv"])

def load_large_benchmark(name: str, root: str):
    if name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root, transform=T.ToUndirected())
        data = dataset[0]
        # OGB arxiv node features and labels
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        
        # Create masks
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[val_idx] = True
        
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True
        
        data.y = data.y.squeeze()
        
        return data, dataset.num_classes, dataset.num_features
        
    raise NotImplementedError(
        f"Dataset {name!r} is not supported."
    )
