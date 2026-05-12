"""
Neighbor-sampled GNN training for large graphs. Not used when MINIBATCH_DATASETS is empty.
"""


def train_gnn_minibatch(*args, **kwargs):
    raise NotImplementedError(
        "Minibatch GNN training is unavailable in this checkout (no large-graph pipeline)."
    )


def infer_logits_minibatch(*args, **kwargs):
    raise NotImplementedError(
        "Minibatch inference is unavailable in this checkout (no large-graph pipeline)."
    )


def train_gnn_dp_minibatch(*args, **kwargs):
    raise NotImplementedError(
        "DP-SGD minibatch training is unavailable in this checkout (no large-graph pipeline)."
    )
