"""
Neighbor-sampled GNN training / inference for large graphs, plus DP-SGD reference.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader


def _neighbor_loader(
    data,
    input_nodes,
    batch_size: int,
    num_neighbors: Optional[Sequence[int]],
    shuffle: bool,
):
    nn = list(num_neighbors) if num_neighbors is not None else [15, 10]
    return NeighborLoader(
        data,
        num_neighbors=nn,
        batch_size=int(batch_size),
        input_nodes=input_nodes,
        shuffle=shuffle,
    )


def train_gnn_minibatch(
    model,
    data,
    device,
    edge_index,
    epochs=50,
    lr=0.01,
    weight_decay=5e-4,
    batch_size=1024,
    num_neighbors=None,
    early_stop_patience=None,
    label_smoothing=0.0,
    dropedge_rate=0.0,
    val_mask=None,
):
    """Neighbor-sampled CE training. Uses data.edge_index (edge_index arg kept for API)."""
    del edge_index  # NeighborLoader reads data.edge_index
    data = data.to("cpu")  # NeighborLoader samples on CPU; move batches to device
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = _neighbor_loader(
        data, data.train_mask, batch_size, num_neighbors, shuffle=True
    )
    num_classes = int(data.y.max().item()) + 1
    best_loss = 1e9
    patience_count = 0
    best_state = None

    for _epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index)
            # NeighborLoader: first batch.batch_size nodes are the seeds.
            seed = out[: batch.batch_size]
            y = batch.y[: batch.batch_size]
            if label_smoothing and float(label_smoothing) > 0:
                log_p = F.log_softmax(seed, 1)
                ls = float(label_smoothing)
                smooth = torch.full_like(log_p, ls / num_classes)
                smooth.scatter_(1, y.view(-1, 1), 1.0 - ls + ls / num_classes)
                loss = -(smooth * log_p).sum(1).mean()
            else:
                loss = F.cross_entropy(seed, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        if early_stop_patience is not None and val_mask is not None and bool(val_mask.any()):
            monitor = total_loss / max(1, n_batches)
            if monitor < best_loss - 1e-4:
                best_loss = monitor
                patience_count = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
            if patience_count >= int(early_stop_patience) and best_state is not None:
                model.load_state_dict(best_state)
                break

    return model


@torch.no_grad()
def infer_logits_minibatch(
    model,
    data,
    edge_index,
    device,
    num_neighbors,
    batch_size,
    num_nodes,
    num_classes,
):
    """Batched neighbor-sampled inference for all nodes; returns CPU logits [N, C]."""
    del edge_index
    data = data.to("cpu")
    model = model.to(device)
    model.eval()
    loader = _neighbor_loader(
        data,
        torch.arange(int(num_nodes)),
        batch_size,
        num_neighbors,
        shuffle=False,
    )
    logits = torch.empty((int(num_nodes), int(num_classes)), dtype=torch.float32)
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        seed = out[: batch.batch_size].cpu()
        # batch.n_id[:batch_size] are global seed ids in recent PyG
        if hasattr(batch, "n_id"):
            ids = batch.n_id[: batch.batch_size].cpu()
        else:
            ids = batch.input_id.cpu() if hasattr(batch, "input_id") else None
            if ids is None:
                raise RuntimeError("NeighborLoader batch missing n_id/input_id")
        logits[ids] = seed
    return logits


def measure_api_qps(model, data, device, num_neighbors, batch_size, warmup=2, timed=5):
    """Rough defended-API query throughput: seed nodes inferred per second."""
    data = data.to("cpu")
    model = model.to(device).eval()
    n = min(int(data.num_nodes), 8192)
    seeds = torch.arange(n)
    loader = _neighbor_loader(data, seeds, batch_size, num_neighbors, shuffle=False)
    # Warmup
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        _ = model(batch.x, batch.edge_index)
        if i + 1 >= warmup:
            break
    loader = _neighbor_loader(data, seeds, batch_size, num_neighbors, shuffle=False)
    t0 = time.time()
    seen = 0
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        _ = model(batch.x, batch.edge_index)
        seen += int(batch.batch_size)
        if i + 1 >= timed:
            break
    elapsed = max(time.time() - t0, 1e-6)
    return float(seen / elapsed)


def _clip_grad_norm_(parameters, max_norm: float) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total = torch.norm(torch.stack([p.grad.detach().norm(2) for p in params]), 2)
    clip_coef = max_norm / (total + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.detach().mul_(clip_coef)
    return float(total.item())


def train_gnn_dp_fullbatch(
    model,
    data,
    device,
    epochs=30,
    lr=0.05,
    weight_decay=0.0,
    max_grad_norm=1.0,
    noise_multiplier=1.0,
    delta=1e-5,
):
    """Full-batch DP-SGD reference (Pareto anchor)."""
    data = data.to(device)
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        _clip_grad_norm_(model.parameters(), float(max_grad_norm))
        for p in model.parameters():
            if p.grad is None:
                continue
            noise = torch.normal(
                mean=0.0,
                std=float(noise_multiplier) * float(max_grad_norm),
                size=p.grad.shape,
                device=p.grad.device,
            )
            p.grad.add_(noise)
        opt.step()
    t = max(1, int(epochs))
    eps = (
        (1.0 / max(float(noise_multiplier), 1e-6))
        * math.sqrt(2.0 * math.log(1.25 / float(delta)))
        * math.sqrt(t)
    )
    return model, float(eps)


def train_gnn_dp_minibatch(
    model,
    data,
    device,
    edge_index,
    epochs=20,
    lr=0.05,
    weight_decay=0.0,
    batch_size=512,
    num_neighbors=None,
    max_grad_norm=1.0,
    noise_multiplier=1.0,
    delta=1e-5,
    dropedge_rate=0.0,
):
    """
    Tunable DP: full-batch on ≤30k nodes; neighbor-sampled DP-SGD on larger graphs.
    noise_multiplier ↓ → weaker privacy / higher utility (ε larger).
    """
    del dropedge_rate
    if getattr(data, "num_nodes", 0) and data.num_nodes <= 30000:
        return train_gnn_dp_fullbatch(
            model,
            data,
            device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            delta=delta,
        )

    data_cpu = data.to("cpu")
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = _neighbor_loader(
        data_cpu, data_cpu.train_mask, batch_size, num_neighbors, shuffle=True
    )
    steps = 0
    model.train()
    for _ in range(int(epochs)):
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index)
            seed = out[: batch.batch_size]
            loss = F.cross_entropy(seed, batch.y[: batch.batch_size])
            loss.backward()
            _clip_grad_norm_(model.parameters(), float(max_grad_norm))
            for p in model.parameters():
                if p.grad is None:
                    continue
                p.grad.add_(
                    torch.normal(
                        0.0,
                        float(noise_multiplier) * float(max_grad_norm),
                        size=p.grad.shape,
                        device=p.grad.device,
                    )
                )
            opt.step()
            steps += 1
    eps = (
        (1.0 / max(float(noise_multiplier), 1e-6))
        * math.sqrt(2.0 * math.log(1.25 / float(delta)))
        * math.sqrt(max(1, steps))
    )
    return model, float(eps)
