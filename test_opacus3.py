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

from opacus.validators import ModuleValidator
print(ModuleValidator.validate(model))

try:
    model, optimizer, _ = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=None,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        poisson_sampling=False, # to avoid needing sample_rate in some versions, or maybe we need to provide it
    )
except Exception as e:
    print(e)
    
