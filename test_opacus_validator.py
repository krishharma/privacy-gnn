import torch
from torch_geometric.nn import GCNConv
from opacus.validators import ModuleValidator

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GCNConv(10, 10)
        self.c2 = GCNConv(10, 10)

model = GCN()
print("Valid:", ModuleValidator.is_valid(model))
print("Errors:", ModuleValidator.validate(model, strict=False))
model = ModuleValidator.fix(model)
print("Valid after fix:", ModuleValidator.is_valid(model))
