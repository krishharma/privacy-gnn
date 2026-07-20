import torch
import numpy as np
from experiment import _load_target_data
from models import GCN
from attacks import label_only_attack
import warnings
warnings.filterwarnings('ignore')

data, num_classes, num_features = _load_target_data("Cora", "data", seed=42, use_official_large=False)
model = GCN(ic=num_features, h=64, oc=num_classes).to("cpu")
model.eval()

trm = data.train_mask.numpy()
tem = data.test_mask.numpy()

auc, acc, dist_m, dist_nm, flip_rate = label_only_attack(model, data, trm, tem, num_samples=100, max_noise_scale=5.0, steps=20)
print(f"AUC: {auc:.4f}, ACC: {acc:.4f}, FLIP_RATE: {flip_rate:.4f}")
print(f"Mean dist M: {np.mean(dist_m):.4f}, Mean dist NM: {np.mean(dist_nm):.4f}")
