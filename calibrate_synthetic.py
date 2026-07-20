import json
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from data import make_synthetic, get_data_dir
from models import GCN
from training import train_gnn

def main():
    center_stds = [0.5, 1.0, 1.5]
    noise_stds = [1.0, 1.5, 2.0]
    homophily_levels = ["low", "medium", "high"]
    density_levels = ["sparse", "medium", "dense"]
    seeds = [42, 43, 44]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    total_iters = len(center_stds) * len(noise_stds) * len(homophily_levels) * len(density_levels) * len(seeds)
    print(f"Total iterations: {total_iters}")
    
    for center_std, noise_std in itertools.product(center_stds, noise_stds):
        for homo, dens in itertools.product(homophily_levels, density_levels):
            for seed in seeds:
                data, nc, nf = make_synthetic(
                    n=400, nf=50, nc=5, 
                    homo=homo, dens=dens, 
                    seed=seed, 
                    center_std=center_std, 
                    noise_std=noise_std
                )
                
                tem = data.test_mask.numpy()
                trm = data.train_mask.numpy()
                yn = data.y.numpy()
                
                # 1. GCN Normal
                model_gcn = GCN(ic=nf, h=64, oc=nc).to(device)
                train_gnn(model_gcn, data, device, epochs=100, lr=0.01, early_stop_patience=15)
                model_gcn.eval()
                with torch.no_grad():
                    p_gcn = model_gcn(data.x.to(device), data.edge_index.to(device)).argmax(1).cpu().numpy()
                gcn_acc = accuracy_score(yn[tem], p_gcn[tem])
                gcn_f1 = f1_score(yn[tem], p_gcn[tem], average="macro")
                
                # 2. MLP Normal
                clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=seed)
                clf.fit(data.x.numpy()[trm], yn[trm])
                p_mlp = clf.predict(data.x.numpy())
                mlp_acc = accuracy_score(yn[tem], p_mlp[tem])
                mlp_f1 = f1_score(yn[tem], p_mlp[tem], average="macro")
                
                # 3. GCN Structure-only (x_v = 1)
                data_struct = data.clone()
                data_struct.x = torch.ones_like(data.x)
                model_struct = GCN(ic=nf, h=64, oc=nc).to(device)
                train_gnn(model_struct, data_struct, device, epochs=100, lr=0.01, early_stop_patience=15)
                model_struct.eval()
                with torch.no_grad():
                    p_struct = model_struct(data_struct.x.to(device), data_struct.edge_index.to(device)).argmax(1).cpu().numpy()
                struct_acc = accuracy_score(yn[tem], p_struct[tem])
                struct_f1 = f1_score(yn[tem], p_struct[tem], average="macro")
                
                # Random baseline
                random_acc = 1.0 / nc
                
                results.append({
                    "center_std": center_std,
                    "noise_std": noise_std,
                    "homophily": homo,
                    "density": dens,
                    "seed": seed,
                    "gcn_acc": gcn_acc,
                    "gcn_f1": gcn_f1,
                    "mlp_acc": mlp_acc,
                    "mlp_f1": mlp_f1,
                    "struct_acc": struct_acc,
                    "struct_f1": struct_f1,
                    "feat_cv_score": data.stats.get("feat_cv_score", float('nan')),
                    "realized_homophily": data.stats.get("realized_homophily"),
                    "realized_density": data.stats.get("realized_density"),
                })
                
    df = pd.DataFrame(results)
    df.to_csv("calibration_results.csv", index=False)
    
    # Analyze to pick the best (center_std, noise_std) pair
    # Rules:
    # - No saturation: GNN macro-F1 and acc in 0.65-0.85
    # - Feature signal exists: MLP > random
    # - Topology signal exists: structure-only > random in high-homophily
    # - Graph value: GNN - MLP > 0 in high-homo
    
    agg = df.groupby(["center_std", "noise_std", "homophily", "density"]).mean().reset_index()
    agg.to_csv("calibration_agg.csv", index=False)
    print("Done generating calibration data.")

if __name__ == "__main__":
    main()
