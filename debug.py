import torch
from torch_geometric.data import Data
from models import GCN
from torch_geometric.nn.conv.gcn_conv import gcn_norm

num_nodes, num_features, num_classes = 10, 8, 3
row, col = [], []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j: row.append(i); col.append(j)
edge_index = torch.tensor([row, col], dtype=torch.long)
x = torch.randn(num_nodes, num_features)
data = Data(x=x, edge_index=edge_index)
model = GCN(num_features, 16, num_classes)
model.eval()
torch.manual_seed(42)
out1 = model.c1(data.x, data.edge_index)

edge_index_norm, edge_weight_norm = gcn_norm(
    data.edge_index, None, num_nodes, improved=False, add_self_loops=True, dtype=data.x.dtype
)
old_normalize = model.c1.normalize
model.c1.normalize = False
out2 = model.c1(data.x, edge_index_norm, edge_weight_norm)
model.c1.normalize = old_normalize

print(torch.abs(out1 - out2).max().item())
