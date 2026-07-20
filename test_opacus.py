import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from opacus import PrivacyEngine

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=16,
    input_nodes=data.train_mask
)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

engine = PrivacyEngine()
try:
    model, optimizer, loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )
    print("Opacus works with NeighborLoader!")
except Exception as e:
    print(f"Opacus failed: {e}")
