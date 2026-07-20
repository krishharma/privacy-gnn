import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from models import GCN
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

model = GCN(10, 16, 2)
optimizer = torch.optim.Adam(model.parameters())

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 10)
y = torch.tensor([0, 1, 0])
data = Data(x=x, edge_index=edge_index, y=y)

loader = NeighborLoader(data, num_neighbors=[2, 2], batch_size=2, input_nodes=torch.tensor([0, 1]))

engine = PrivacyEngine()
model, optimizer, loader = engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0
)

for batch in loader:
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index)
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
    loss.backward()
    optimizer.step()

print("Backward pass successful!")
