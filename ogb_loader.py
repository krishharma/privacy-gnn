"""
Large-graph loaders (OGB, etc.). This checkout supports Cora/Citeseer/Planetoid and synthetics.
Enable ogbn-arxiv by installing ogb and extending load_large_benchmark.
"""
MINIBATCH_DATASETS = frozenset()


def load_large_benchmark(name: str, root: str):
    raise NotImplementedError(
        f"Dataset {name!r} requires OGB/minibatch support not bundled in this repo snapshot. "
        "Use datasets in experiment_config_paper.yaml (Cora, Citeseer, synthetics only)."
    )
