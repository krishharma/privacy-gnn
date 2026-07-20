import torch
import time
from ogb_loader import load_large_benchmark
from graph_minibatch import CustomKhopLoader

data, _, _ = load_large_benchmark("ogbn-arxiv", "data")
loader = CustomKhopLoader(data, input_nodes=torch.where(data.train_mask)[0], batch_size=256, num_hops=2)

t0 = time.time()
for i, batch in enumerate(loader):
    print(f"Batch {i} ({time.time()-t0:.1f}s): target nodes: {batch.batch_size}, total nodes in subgraph: {batch.x.size(0)}")
    t0 = time.time()
    if i >= 5:
        break
