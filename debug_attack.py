import torch
import numpy as np
from data import get_dataset
from models import GCN
from attacks import label_only_attack

data, num_classes, num_features = get_dataset("Cora", "data", "cpu", False)
model = GCN(ic=num_features, h=64, oc=num_classes).to("cpu")
trm = data.train_mask.numpy()
tem = data.test_mask.numpy()

try:
    auc, acc = label_only_attack(model, data, trm, tem, num_samples=10, max_noise_scale=2.0, steps=5)
    print("SUCCESS", auc, acc)
except Exception as e:
    import traceback
    traceback.print_exc()
