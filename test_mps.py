import torch
from torch_geometric.nn import GCNConv

device = torch.device('mps')
print("MPS available:", torch.backends.mps.is_available())

x = torch.randn(169000, 128).to(device)
edge_index = torch.randint(0, 169000, (2, 2300000)).to(device)
conv = GCNConv(128, 64).to(device)

import time
t0 = time.time()
for _ in range(50):
    out = conv(x, edge_index)
    loss = out.sum()
    loss.backward()
print(f"Done in {time.time()-t0:.2f}s")
