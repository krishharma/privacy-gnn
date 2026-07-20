import pandas as pd

df = pd.read_csv("calibration_agg.csv")
print(df.groupby(["center_std", "noise_std"]).mean(numeric_only=True)[["gcn_acc", "mlp_acc", "struct_acc"]])

for center in [0.5, 1.0, 1.5]:
    for noise in [1.0, 1.5, 2.0]:
        print(f"\n--- center_std={center}, noise_std={noise} ---")
        sub = df[(df.center_std == center) & (df.noise_std == noise)]
        print(sub[["homophily", "density", "gcn_acc", "mlp_acc", "struct_acc", "gcn_f1"]])
