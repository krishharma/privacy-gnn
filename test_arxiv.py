import time
from ogb_loader import load_large_benchmark
from data import get_data_dir
from models import GCN
from graph_minibatch import train_gnn_minibatch
import torch

t0 = time.time()
data, nc, nf = load_large_benchmark("ogbn-arxiv", get_data_dir() + "/ogb")
print(f"Loaded in {time.time()-t0:.2f}s")
model = GCN(ic=nf, h=64, oc=nc)
t1 = time.time()
train_gnn_minibatch(model, data, torch.device('cpu'), data.edge_index, epochs=2, lr=0.01, batch_size=1024, num_neighbors=[15,10])
print(f"Trained 2 epochs in {time.time()-t1:.2f}s")
