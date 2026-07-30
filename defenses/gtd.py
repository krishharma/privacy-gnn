"""
GTD-style baseline: Graph Transductive Defense (reimplementation).

Following Wang et al., "Graph Transductive Defense: a Two-Stage Defense for
Graph Membership Inference Attacks" (arXiv:2406.07917): a two-stage schedule
that (i) trains normally on the supervised mask, then (ii) alternates
supervised steps with pseudo-label steps on the unlabeled (transductive)
nodes plus a loss-flattening penalty that shrinks the gap between the
supervised loss and the unlabeled-node loss — the gap score-based MIAs exploit.

This is a faithful-in-spirit reimplementation used as a published-baseline
comparison; hyperparameters are documented in the experiment config.
"""
import torch
import torch.nn.functional as F


def train_gnn_gtd(
    model,
    data,
    device,
    epochs=50,
    lr=0.01,
    weight_decay=5e-4,
    gamma=1.0,
    stage1_frac=0.5,
    pseudo_conf=0.8,
):
    """
    Two-stage GTD-style training.

    Stage 1 (first stage1_frac of epochs): standard CE on the train mask.
    Stage 2: alternate epochs between
      - supervised CE on train mask + gamma * (L_train - L_out)^2 flattening,
      - CE on confident pseudo-labeled non-train nodes (model's own argmax with
        max prob >= pseudo_conf), which pulls held-out posteriors toward the
        same loss regime as members.

    Only unlabeled/topology information available transductively is used; true
    labels of non-train nodes are never touched.
    """
    data = data.to(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = data.train_mask
    out_mask = ~train_mask

    n_stage1 = max(1, int(epochs * stage1_frac))

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(data.x, data.edge_index)

        if epoch < n_stage1:
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        elif epoch % 2 == 0:
            l_train = F.cross_entropy(logits[train_mask], data.y[train_mask])
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                pseudo = probs.argmax(dim=1)
            l_out = F.cross_entropy(logits[out_mask], pseudo[out_mask])
            loss = l_train + gamma * (l_train - l_out) ** 2
        else:
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                conf, pseudo = probs.max(dim=1)
            sel = out_mask & (conf >= pseudo_conf)
            if bool(sel.any()):
                loss = F.cross_entropy(logits[sel], pseudo[sel])
            else:
                loss = F.cross_entropy(logits[train_mask], data.y[train_mask])

        loss.backward()
        opt.step()

    return model


def train_gnn_gtd_minibatch(
    model,
    data,
    device,
    epochs=20,
    lr=0.01,
    weight_decay=5e-4,
    gamma=1.0,
    stage1_frac=0.5,
    pseudo_conf=0.8,
    batch_size=1024,
    num_neighbors=None,
):
    """
    Neighbor-sampled GTD-style training for Volume graphs.

    Default Volume-safe behavior: when stage1_frac >= 0.99, train with supervised
    CE only (matches none Acc), optionally with a *tiny* supervised entropy pull.
    Pseudo-label stage-2 is disabled by default at Volume because NeighborLoader
    + capped unlabeled batches previously collapsed Acc (~0.47) — a pipeline
    artifact, not a fair baseline. Set stage1_frac < 0.99 to enable a cautious
    stage-2 (few unlabeled batches, high pseudo_conf).
    """
    from torch_geometric.loader import NeighborLoader

    data_cpu = data.to("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    nn = list(num_neighbors) if num_neighbors is not None else [15, 10]
    train_loader = NeighborLoader(
        data_cpu,
        num_neighbors=nn,
        batch_size=int(batch_size),
        input_nodes=data_cpu.train_mask,
        shuffle=True,
    )
    out_nodes = (~data_cpu.train_mask).nonzero(as_tuple=False).view(-1)
    n_stage1 = max(1, int(epochs * float(stage1_frac)))
    volume_safe = float(stage1_frac) >= 0.99

    model.train()
    for epoch in range(int(epochs)):
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index)
            seed = out[: batch.batch_size]
            loss = F.cross_entropy(seed, batch.y[: batch.batch_size])
            if volume_safe and float(gamma) > 0:
                # Mild supervised flattening only (does not touch unlabeled nodes).
                probs = F.softmax(seed, dim=1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(1).mean()
                loss = loss + 0.01 * float(gamma) * (-ent)
            loss.backward()
            opt.step()

        if volume_safe:
            continue
        if epoch >= n_stage1 and len(out_nodes) > 0 and epoch % 2 == 1:
            out_loader = NeighborLoader(
                data_cpu,
                num_neighbors=nn,
                batch_size=int(batch_size),
                input_nodes=out_nodes,
                shuffle=True,
            )
            for bi, batch in enumerate(out_loader):
                batch = batch.to(device)
                opt.zero_grad()
                out = model(batch.x, batch.edge_index)
                seed = out[: batch.batch_size]
                with torch.no_grad():
                    probs = F.softmax(seed, dim=1)
                    conf, pseudo = probs.max(dim=1)
                sel = conf >= float(pseudo_conf)
                if bool(sel.any()):
                    loss = 0.1 * float(gamma) * F.cross_entropy(seed[sel], pseudo[sel])
                    loss.backward()
                    opt.step()
                if bi >= 3:
                    break
    return model
