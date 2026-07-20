import argparse
import sys
from experiment import run_one
from config import load_config

def main():
    print("Running EPSD regression test...")
    config = load_config("experiment_config_broad.yaml")
    
    # 1. Run Baseline (none)
    print("-> Running baseline (none)")
    res_none = run_one(
        dataset_name="Cora",
        model_name="GCN",
        defense_name="none",
        defense_params={},
        seed=42,
        config=config
    )
    
    # 2. Run EPSD
    print("-> Running EPSD (lambda=5.0)")
    res_epsd = run_one(
        dataset_name="Cora",
        model_name="GCN",
        defense_name="epsd",
        defense_params={"lambda_epsd": 5.0, "ablation": "none"},
        seed=42,
        config=config
    )
    
    # 3. Assertions
    print("-> Verifying Hashes...")
    hash_none = res_none['model_sha256']
    hash_epsd = res_epsd['model_sha256']
    pred_none = res_none['prediction_sha256']
    pred_epsd = res_epsd['prediction_sha256']
    
    print(f"   Baseline model hash: {hash_none}")
    print(f"   EPSD model hash:     {hash_epsd}")
    print(f"   Baseline pred hash:  {pred_none}")
    print(f"   EPSD pred hash:      {pred_epsd}")
    
    assert hash_none != "N/A" and hash_epsd != "N/A", "Hashes missing!"
    assert hash_none != hash_epsd, "Model hashes are identical! EPSD is still a no-op."
    assert pred_none != pred_epsd, "Prediction hashes are identical! EPSD is still a no-op."
    
    print("-> Verifying KL Divergence Tracking...")
    kl_ep1 = res_epsd['epsd_kl_loss_ep1']
    kl_final = res_epsd['epsd_kl_loss_final']
    
    print(f"   EPSD KL Epoch 1: {kl_ep1}")
    print(f"   EPSD KL Final:   {kl_final}")
    
    import math
    assert not math.isnan(kl_ep1), "KL Loss Ep 1 is NaN"
    assert kl_ep1 > 1e-8, f"KL Loss Ep 1 is {kl_ep1}, should be > 1e-8"
    
    print("Regression test passed successfully!")

if __name__ == "__main__":
    main()
