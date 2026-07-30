"""
SAMI: Structure-Aware Membership Indistinguishability defense.

Three coupled modules:
1. LTE (Leakage Topology Estimator): per-node structural leakage risk r_v in [0, 1]
   computed from local degree, train-neighbor label homophily, and supervised-neighbor
   fraction. Motivated by the empirical finding that GCN membership leakage
   concentrates in low-homophily, sparse neighborhoods.
2. Risk-weighted phi-indistinguishability loss: a small membership discriminator D is
   trained on the 4-D phi attack-feature map (max prob, true-label prob, entropy,
   modified entropy); the GNN is simultaneously trained so that D cannot separate
   members from the defender's own held-out (validation) nodes, with per-node risk
   weights r_v concentrating the defense where topology creates the most signal.
3. HCAG (Homophily-Conditioned Aggregation Gate): optional gated message passing
   (see models.GatedGCN / models.GatedSAGE) that down-weights aggregation paths
   with high structural risk.

Privacy hygiene: the adversarial alignment uses TRAIN vs VALIDATION nodes only.
Test nodes are never used by the defense, so the final membership evaluation
(train members vs test non-members) is uncontaminated.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-10


# ----------------------------------------------------------------------------
# Module 1: LTE — Leakage Topology Estimator
# ----------------------------------------------------------------------------

def compute_lte_risk(data, uniform=False, arch="sage", arch_aware=True):
    """
    Per-node structural leakage risk r_v in [0, 1].

    Components (all computable from the graph + train mask, no test information):
      - inverse-degree risk: sparse neighborhoods leak more
      - heterophily risk: 1 - (label agreement among *train* neighbors)
      - supervision risk: fraction of neighbors carrying supervised labels
      - (arch-aware) degree-heterophily interaction upweighted for GCN, which
        preserves a larger φ-gap under heterophilic sparsity than GraphSAGE

    Returns a torch.FloatTensor of shape [n], min-max normalized to [0, 1].
    If uniform=True, returns all-ones (ablation: -LTE).
    """
    n = data.num_nodes
    if uniform:
        return torch.ones(n, dtype=torch.float)

    ei = data.edge_index
    y = data.y
    train_mask = data.train_mask

    deg = torch.zeros(n, dtype=torch.float)
    if ei.size(1) > 0:
        deg.index_add_(0, ei[0], torch.ones(ei.size(1), dtype=torch.float))

    inv_deg_risk = 1.0 / (1.0 + deg)

    same_label = torch.zeros(n, dtype=torch.float)
    train_nbrs = torch.zeros(n, dtype=torch.float)
    if ei.size(1) > 0:
        src, dst = ei[0], ei[1]
        nbr_is_train = train_mask[dst].float()
        train_nbrs.index_add_(0, src, nbr_is_train)
        agree = ((y[src] == y[dst]).float()) * nbr_is_train
        same_label.index_add_(0, src, agree)

    homophily_local = same_label / train_nbrs.clamp(min=1.0)
    heterophily_risk = 1.0 - homophily_local
    heterophily_risk[train_nbrs == 0] = 0.5

    supervision_risk = train_nbrs / deg.clamp(min=1.0)

    if arch_aware and str(arch).lower() in ("gcn",):
        # GCN amplifies structure-conditioned leakage: overweight sparse×heterophilic.
        arch_term = inv_deg_risk * heterophily_risk
        risk = (inv_deg_risk + heterophily_risk + supervision_risk + arch_term) / 4.0
    else:
        risk = (inv_deg_risk + heterophily_risk + supervision_risk) / 3.0
    lo, hi = risk.min(), risk.max()
    if (hi - lo) > 1e-8:
        risk = (risk - lo) / (hi - lo)
    else:
        risk = torch.ones_like(risk)
    return risk


def allocate_risk_budget(risk, budget_B: float, base_scale: float = 0.35):
    """
    Finite release budget: choose per-node noise scales σ_v ∝ r_v so that
    sum_v r_v σ_v = B (when B>0). Returns per-node scales (numpy).
    If B<=0, falls back to base_scale * r_v.
    """
    r = np.asarray(risk, dtype=float).reshape(-1)
    if budget_B is None or float(budget_B) <= 0:
        return base_scale * r
    mass = float(r.sum())
    if mass < 1e-12:
        return np.full_like(r, float(base_scale))
    # σ_v = B * r_v / sum(r)  ⇒  sum r_v σ_v = B * sum(r^2)/sum(r); instead set
    # σ_v = c * r_v with c chosen so sum(r_v * σ_v) = B ⇒ c * ||r||_2^2 = B.
    denom = float(np.dot(r, r))
    c = float(budget_B) / max(denom, 1e-12)
    return c * r


# ----------------------------------------------------------------------------
# Module 2: differentiable phi features + membership discriminator
# ----------------------------------------------------------------------------

def phi_features_torch(probs, labels):
    """
    Differentiable 4-D phi attack-feature map (same map as attacks.extract_features):
      phi1 = max_c p_c
      phi2 = p_{y}
      phi3 = entropy(p)
      phi4 = -(1 - phi1) log(phi1)
    probs: [m, C] softmax outputs; labels: [m] long.
    """
    phi1 = probs.max(dim=1).values
    phi2 = probs.gather(1, labels.view(-1, 1)).squeeze(1)
    phi3 = -(probs * torch.log(probs + EPS)).sum(dim=1)
    phi4 = -(1.0 - phi1) * torch.log(phi1 + EPS)
    return torch.stack([phi1, phi2, phi3, phi4], dim=1)


class MembershipDiscriminator(nn.Module):
    """Small MLP on the 4-D phi map predicting membership (logit output)."""

    def __init__(self, in_dim=4, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, phi):
        return self.net(phi).squeeze(-1)


# ----------------------------------------------------------------------------
# Optional inference stage: risk-scaled temperature
# ----------------------------------------------------------------------------

def risk_scaled_temperature(logits, risk, beta=0.0):
    """
    Per-node temperature T_v = 1 + beta * r_v applied to logits before softmax.
    beta=0 disables the transform.
    """
    if beta <= 0:
        return logits
    t = (1.0 + beta * risk.to(logits.device)).unsqueeze(1)
    return logits / t


def risk_scaled_posterior_noise(p, risk, scale=0.0, seed=0):
    """
    Structure-aware Laplace noise on posteriors: row v gets Laplace(0, scale * r_v).
    When scale=0, returns p unchanged. Renormalizes after clipping.
    """
    import numpy as np

    if scale <= 0:
        return p
    p = np.array(p, dtype=float, copy=True)
    r = np.asarray(risk, dtype=float)
    rng = np.random.RandomState(int(seed))
    noise = rng.laplace(0.0, 1.0, size=p.shape) * (scale * r)[:, None]
    p = np.clip(p + noise, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    uniform = np.full((1, p.shape[1]), 1.0 / p.shape[1])
    p = np.where(s > 0, p / np.clip(s, 1e-12, None), uniform)
    return p


# ----------------------------------------------------------------------------
# SAMI training loop (full-batch, transductive)
# ----------------------------------------------------------------------------

def train_gnn_sami(
    model,
    data,
    device,
    epochs=50,
    lr=0.01,
    weight_decay=5e-4,
    lam=0.5,
    use_lte=True,
    use_gate=True,
    disc_lr=0.01,
    disc_steps=2,
    warmup_epochs=5,
    risk=None,
    entropy_coef=0.05,
    arch="sage",
    arch_aware=True,
):
    """
    SAMI training:
      L = L_CE(train)
        + lam * risk-weighted MMD(phi_train, phi_val)   # align attack features
        + lam * AdvReg(fool D toward non-member)        # adversarial indistinguishability
        + entropy_coef * mean_{v in train} r_v H(p_v)   # calm high-risk overconfidence

    Non-member proxies = validation nodes when available (never test).
    Returns (model, risk_tensor).
    """
    data = data.to(device)
    model = model.to(device)

    if risk is None:
        risk = compute_lte_risk(
            data.cpu() if data.x.is_cuda else data,
            uniform=not use_lte,
            arch=arch,
            arch_aware=arch_aware,
        )
    risk = risk.to(device)

    disc = MembershipDiscriminator().to(device)
    opt_g = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_d = torch.optim.Adam(disc.parameters(), lr=disc_lr)

    train_mask = data.train_mask
    val_mask = getattr(data, "val_mask", None)
    if val_mask is not None and bool(val_mask.any()):
        proxy_mask = val_mask
    else:
        proxy_mask = ~train_mask

    train_idx = train_mask.nonzero(as_tuple=True)[0]
    proxy_idx = proxy_mask.nonzero(as_tuple=True)[0]
    n_pair = min(len(train_idx), len(proxy_idx))

    supports_gate = use_gate and hasattr(model, "supports_risk") and model.supports_risk

    def forward_model():
        if supports_gate:
            return model(data.x, data.edge_index, risk=risk)
        return model(data.x, data.edge_index)

    model.train()
    for epoch in range(epochs):
        # --- (1) discriminator step(s) on detached phi ---
        if lam > 0 and epoch >= warmup_epochs and n_pair >= 2:
            for _ in range(disc_steps):
                with torch.no_grad():
                    logits = forward_model()
                    probs = F.softmax(logits, dim=1)
                perm_t = train_idx[torch.randperm(len(train_idx), device=device)[:n_pair]]
                perm_p = proxy_idx[torch.randperm(len(proxy_idx), device=device)[:n_pair]]
                idx = torch.cat([perm_t, perm_p])
                m = torch.cat(
                    [torch.ones(n_pair, device=device), torch.zeros(n_pair, device=device)]
                )
                phi = phi_features_torch(probs[idx], data.y[idx]).detach()
                opt_d.zero_grad()
                d_loss = F.binary_cross_entropy_with_logits(disc(phi), m)
                d_loss.backward()
                opt_d.step()

        # --- (2) generator (GNN) step ---
        opt_g.zero_grad()
        logits = forward_model()
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])

        if lam > 0 and epoch >= warmup_epochs and n_pair >= 2:
            probs = F.softmax(logits, dim=1)
            n_t = min(len(train_idx), max(n_pair, len(train_idx) // 2))
            n_p = min(len(proxy_idx), max(n_pair, len(proxy_idx) // 2))
            perm_t = train_idx[torch.randperm(len(train_idx), device=device)[:n_t]]
            perm_p = proxy_idx[torch.randperm(len(proxy_idx), device=device)[:n_p]]
            idx = torch.cat([perm_t, perm_p])

            phi = phi_features_torch(probs[idx], data.y[idx])
            d_out = disc(phi)
            nonmember_target = torch.zeros_like(d_out)
            per_node = F.binary_cross_entropy_with_logits(
                d_out, nonmember_target, reduction="none"
            )

            phi_t = phi_features_torch(probs[perm_t], data.y[perm_t])
            phi_p = phi_features_torch(probs[perm_p], data.y[perm_p])
            w = risk[perm_t]
            w = w / w.sum().clamp(min=1e-8)
            mean_t = (phi_t * w.unsqueeze(1)).sum(0)
            mean_p = phi_p.mean(0)
            # Also match second moments (variance) for stronger alignment.
            var_t = ((phi_t - mean_t) ** 2 * w.unsqueeze(1)).sum(0)
            var_p = ((phi_p - mean_p) ** 2).mean(0)
            mmd = ((mean_t - mean_p) ** 2).sum() + 0.5 * ((var_t - var_p) ** 2).sum()

            # Entropy regularizer on high-risk training nodes (reduce overconfidence).
            ent = -(probs[perm_t] * torch.log(probs[perm_t] + EPS)).sum(1)
            ent_loss = (risk[perm_t] * (-ent)).mean()  # maximize entropy → minimize -H

            loss = (
                loss
                + lam * (risk[idx] * per_node).mean()
                + lam * mmd
                + entropy_coef * ent_loss
            )

        loss.backward()
        opt_g.step()

    return model, risk.detach().cpu()


def train_gnn_sami_minibatch(
    model,
    data,
    device,
    epochs=20,
    lr=0.01,
    weight_decay=5e-4,
    lam=0.5,
    use_lte=True,
    warmup_epochs=3,
    entropy_coef=0.05,
    batch_size=1024,
    num_neighbors=None,
    arch="sage",
    arch_aware=True,
    align_every=2,
    align_nodes=2048,
):
    """
    Volume-scale SAMI: full-graph LTE once; NeighborLoader CE; periodic
    risk-weighted AdvReg+MMD on a subsample of train/val seeds (HCAG off).
    """
    from torch_geometric.loader import NeighborLoader

    data_cpu = data.to("cpu")
    model = model.to(device)
    risk = compute_lte_risk(
        data_cpu, uniform=not use_lte, arch=arch, arch_aware=arch_aware
    )
    risk_dev = risk.to(device)

    disc = MembershipDiscriminator().to(device)
    opt_g = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_d = torch.optim.Adam(disc.parameters(), lr=0.01)

    nn = list(num_neighbors) if num_neighbors is not None else [15, 10]
    train_loader = NeighborLoader(
        data_cpu,
        num_neighbors=nn,
        batch_size=int(batch_size),
        input_nodes=data_cpu.train_mask,
        shuffle=True,
    )
    train_idx = data_cpu.train_mask.nonzero(as_tuple=False).view(-1)
    val_mask = getattr(data_cpu, "val_mask", None)
    if val_mask is not None and bool(val_mask.any()):
        proxy_idx = val_mask.nonzero(as_tuple=False).view(-1)
    else:
        proxy_idx = (~data_cpu.train_mask).nonzero(as_tuple=False).view(-1)

    model.train()
    for epoch in range(int(epochs)):
        for batch in train_loader:
            batch = batch.to(device)
            opt_g.zero_grad()
            out = model(batch.x, batch.edge_index)
            seed = out[: batch.batch_size]
            y = batch.y[: batch.batch_size]
            loss = F.cross_entropy(seed, y)
            if entropy_coef > 0 and use_lte and hasattr(batch, "n_id"):
                ids = batch.n_id[: batch.batch_size]
                r = risk_dev[ids]
                probs = F.softmax(seed, dim=1)
                ent = -(probs * torch.log(probs + EPS)).sum(1)
                loss = loss + float(entropy_coef) * (r * (-ent)).mean()
            loss.backward()
            opt_g.step()

        # Periodic φ-alignment on subsampled seeds (never uses test).
        if lam > 0 and epoch >= int(warmup_epochs) and epoch % max(1, int(align_every)) == 0:
            n_t = min(int(align_nodes), len(train_idx))
            n_p = min(int(align_nodes), len(proxy_idx))
            if n_t < 2 or n_p < 2:
                continue
            perm_t = train_idx[torch.randperm(len(train_idx))[:n_t]]
            perm_p = proxy_idx[torch.randperm(len(proxy_idx))[:n_p]]
            align_loader = NeighborLoader(
                data_cpu,
                num_neighbors=nn,
                batch_size=min(512, n_t + n_p),
                input_nodes=torch.cat([perm_t, perm_p]),
                shuffle=False,
            )
            # Collect seed logits for the subsample
            id_to_logit = {}
            model.eval()
            with torch.no_grad():
                for batch in align_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index)
                    seed = out[: batch.batch_size].cpu()
                    ids = batch.n_id[: batch.batch_size].cpu()
                    for i, gid in enumerate(ids.tolist()):
                        id_to_logit[gid] = seed[i]
            model.train()
            t_ids = [int(i) for i in perm_t.tolist() if int(i) in id_to_logit]
            p_ids = [int(i) for i in perm_p.tolist() if int(i) in id_to_logit]
            n_pair = min(len(t_ids), len(p_ids))
            if n_pair < 2:
                continue
            t_ids, p_ids = t_ids[:n_pair], p_ids[:n_pair]
            logits = torch.stack(
                [id_to_logit[i] for i in t_ids] + [id_to_logit[i] for i in p_ids]
            ).to(device)
            labels = torch.tensor(
                [int(data_cpu.y[i]) for i in t_ids + p_ids], device=device
            )
            probs = F.softmax(logits, dim=1)
            phi = phi_features_torch(probs, labels).detach()
            m = torch.cat(
                [torch.ones(n_pair, device=device), torch.zeros(n_pair, device=device)]
            )
            opt_d.zero_grad()
            d_loss = F.binary_cross_entropy_with_logits(disc(phi), m)
            d_loss.backward()
            opt_d.step()

            # One generator alignment step via a fresh CE+align minibatch on train seeds
            opt_g.zero_grad()
            # Re-forward train seeds only for AdvReg (cheap NeighborLoader batch)
            t_loader = NeighborLoader(
                data_cpu,
                num_neighbors=nn,
                batch_size=min(512, n_pair),
                input_nodes=torch.tensor(t_ids, dtype=torch.long),
                shuffle=False,
            )
            batch = next(iter(t_loader)).to(device)
            out = model(batch.x, batch.edge_index)
            seed = out[: batch.batch_size]
            y = batch.y[: batch.batch_size]
            loss = F.cross_entropy(seed, y)
            probs = F.softmax(seed, dim=1)
            phi = phi_features_torch(probs, y)
            d_out = disc(phi)
            per_node = F.binary_cross_entropy_with_logits(
                d_out, torch.zeros_like(d_out), reduction="none"
            )
            ids = batch.n_id[: batch.batch_size]
            r = risk_dev[ids]
            loss = loss + float(lam) * (r * per_node).mean()
            loss.backward()
            opt_g.step()

    return model, risk.detach().cpu()
