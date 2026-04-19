import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from sae_lens import SAE
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from utils import get_activations, interpolate, get_logits, get_sae_info

# Configuration
MODEL_NAME = "gpt2-small"
LAYER = 6
SAE_ID = f"blocks.{LAYER}.hook_resid_pre"
SAE_RELEASE = "gpt2-small-res-jb"

# Load model
model = HookedTransformer.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define Concept Pairs
exp_configs = [
    {
        "name": "Congruent (Paris-Lyon)",
        "prompt": "The city of",
        "target1": " Paris",
        "target2": " Lyon"
    },
    {
        "name": "Exclusive (Paris-London)",
        "prompt": "The capital of France is",
        "target1": " Paris",
        "target2": " London"
    },
    {
        "name": "Unrelated (Paris-Justice)",
        "prompt": "The word is",
        "target1": " Paris",
        "target2": " justice"
    }
]

# Load SAE
print(f"\nAttempting to load SAE {SAE_ID} from {SAE_RELEASE}...")
try:
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID
    )
    sae = sae.to(device)
    print("SAE loaded successfully.")
except Exception as e:
    print(f"SAE loading failed: {e}")
    sae = None

# Results
all_results = []

for config in exp_configs:
    name = config["name"]
    prompt = config["prompt"]
    t1 = config["target1"]
    t2 = config["target2"]
    
    print(f"\nProcessing {name}...")
    
    # Extract activations at hook_resid_pre
    # Note: utils.get_activations uses hook_resid_post by default. 
    # I'll modify it here or just call model directly.
    
    def get_act(text):
        _, cache = model.run_with_cache(text)
        return cache[SAE_ID][0, -1, :]

    v1 = get_act(f"{prompt}{t1}")
    v2 = get_act(f"{prompt}{t2}")
    
    # Interpolate
    steps = 21
    v_interp, alphas = interpolate(v1, v2, steps=steps)
    v_interp = v_interp.to(device)
    
    # Logit Lens
    logits = get_logits(model, v_interp)
    probs = F.softmax(logits, dim=-1)
    
    idx1 = model.to_single_token(t1)
    idx2 = model.to_single_token(t2)
    
    p1 = probs[:, idx1].detach().cpu().numpy()
    p2 = probs[:, idx2].detach().cpu().numpy()
    
    # SAE analysis
    if sae is not None:
        feature_acts, mse, l0 = get_sae_info(sae, v_interp)
        mse = mse.detach().cpu().numpy()
        l0 = l0.detach().cpu().numpy()
        # Top features at middle of interpolation (alpha=0.5)
        mid_idx = steps // 2
        top_features = torch.topk(feature_acts[mid_idx], k=5)
        top_feat_indices = top_features.indices.detach().cpu().numpy().tolist()
        top_feat_vals = top_features.values.detach().cpu().numpy().tolist()
    else:
        mse = np.zeros(steps)
        l0 = np.zeros(steps)
        top_feat_indices = []
        top_feat_vals = []

    # Store data
    res_df = pd.DataFrame({
        "alpha": alphas,
        "p1": p1,
        "p2": p2,
        "mse": mse,
        "l0": l0
    })
    res_df["experiment"] = name
    all_results.append(res_df)
    
    # Print summary
    print(f"  Alpha=0.5: P1={p1[mid_idx]:.4f}, P2={p2[mid_idx]:.4f}, MSE={mse[mid_idx]:.4e}, L0={l0[mid_idx]}")

# Save all results
full_res_df = pd.concat(all_results)
full_res_df.to_csv("results/interpolation_results.csv", index=False)
print("\nResults saved to results/interpolation_results.csv")

# Plotting
plt.figure(figsize=(15, 10))

# Plot Proportions
plt.subplot(2, 2, 1)
for df in all_results:
    plt.plot(df["alpha"], df["p1"], label=f"{df['experiment'].iloc[0]} (P1)")
plt.title("Probability of Concept 1")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
for df in all_results:
    plt.plot(df["alpha"], df["p2"], label=f"{df['experiment'].iloc[0]} (P2)")
plt.title("Probability of Concept 2")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
for df in all_results:
    plt.plot(df["alpha"], df["mse"], label=df["experiment"].iloc[0])
plt.title("SAE Reconstruction MSE")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
for df in all_results:
    plt.plot(df["alpha"], df["l0"], label=df["experiment"].iloc[0])
plt.title("SAE L0 Sparsity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/interpolation_analysis.png")
print("Analysis plots saved to results/interpolation_analysis.png")

