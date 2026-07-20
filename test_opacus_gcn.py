import torch
from torch_geometric.nn import GCNConv
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset
import traceback

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(10, 10)
    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

model = GCN()
optimizer = torch.optim.Adam(model.parameters())
loader = DataLoader(TensorDataset(torch.arange(10)), batch_size=2)
engine = PrivacyEngine()
model, optimizer, loader = engine.make_private(
    module=model, optimizer=optimizer, data_loader=loader,
    noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=False
)
x = torch.randn(10, 10)
edge_index = torch.tensor([[0, 1], [1, 0]])
try:
    out = model(x, edge_index)
    loss = out.sum()
    loss.backward()
    print("Train step succeeded!")
except Exception as e:
    traceback.print_exc()
