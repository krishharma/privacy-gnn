import pandas as pd
import json

df = pd.read_csv("results/small_graph_comparison_table.csv")
# Filter columns for readability
cols = [
    "dataset", "model", "defense", 
    "test_accuracy", "shadow_attack_auc", "conf_attack_auc", "label_only_attack_auc"
]
df = df[cols]

# Create a nice markdown representation
with open("/Users/krishsharma/.gemini/antigravity-ide/brain/e4e344f3-c312-441d-a0ed-694087a68de2/SMALL_GRAPH_RESULTS.md", "w") as f:
    f.write("# Small & Synthetic Graph Privacy Evaluation\n\n")
    f.write("This table shows the 95% Bootstrap CIs over 10 random seeds for accuracy and various Membership Inference Attacks.\n\n")
    f.write("## Cora & Citeseer (Real Graphs)\n")
    f.write(df[df["dataset"].isin(["Cora", "Citeseer"])].to_markdown(index=False))
    f.write("\n\n## Synthetic Sparse (High vs Low Homophily)\n")
    f.write(df[df["dataset"].str.contains("sparse")].to_markdown(index=False))
    f.write("\n\n## Synthetic Dense (High vs Low Homophily)\n")
    f.write(df[df["dataset"].str.contains("dense")].to_markdown(index=False))

print("Markdown artifact generated!")
