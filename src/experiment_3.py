import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from sae_lens import SAE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import get_activations, interpolate, get_logits, get_sae_info

# Configuration
MODEL_NAME = "gpt2-small"
LAYER = 6
SAE_ID = f"blocks.{LAYER}.hook_resid_pre"
SAE_RELEASE = "gpt2-small-res-jb"

model = HookedTransformer.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

try:
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
    sae = sae.to(device)
except:
    sae = None

def get_act(text):
    _, cache = model.run_with_cache(text)
    return cache[SAE_ID][0, -1, :]

exp_configs = [
    {
        "name": "Related",
        "p1": "The capital of France is",
        "p2": "The capital of Germany is"
    },
    {
        "name": "Incongruent",
        "p1": "The capital of France is",
        "p2": "The favorite color of a dog is"
    }
]

def kl_div(logits1, logits2):
    """Compute KL(P1 || P2) between two logit vectors."""
    # We use full vocab distribution
    p1 = F.softmax(logits1, dim=-1)
    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    return (p1 * (log_p1 - log_p2)).sum(dim=-1)

all_results = []

for config in exp_configs:
    name = config["name"]
    v1 = get_act(config["p1"])
    v2 = get_act(config["p2"])
    
    steps = 41 # Finer steps for metric
    v_interp, alphas = interpolate(v1, v2, steps=steps)
    v_interp = v_interp.to(device)
    
    logits = get_logits(model, v_interp)
    
    # Compute metrics
    entropies = []
    metrics = []
    
    for i in range(steps):
        p = F.softmax(logits[i], dim=-1)
        ent = - (p * torch.log(p + 1e-10)).sum().item()
        entropies.append(ent)
        
        if i < steps - 1:
            # Metric g(alpha) = KL(P(alpha) || P(alpha+delta)) / delta^2
            delta = 1.0 / (steps - 1)
            kl = kl_div(logits[i], logits[i+1]).item()
            metric = kl / (delta**2)
            metrics.append(metric)
        else:
            metrics.append(np.nan)

    if sae:
        feature_acts, mse, l0 = get_sae_info(sae, v_interp)
        mse = mse.detach().cpu().numpy()
        l0 = l0.detach().cpu().numpy()
        
        # Analyze top features at alpha=0.5
        mid_idx = steps // 2
        top_f = torch.topk(feature_acts[mid_idx], k=5)
        print(f"\n{name} Top Features at 0.5: {top_f.indices.tolist()}")
    else:
        mse = np.zeros(steps)
        l0 = np.zeros(steps)

    df = pd.DataFrame({
        "alpha": alphas,
        "entropy": entropies,
        "metric": metrics,
        "mse": mse,
        "l0": l0,
        "experiment": name
    })
    all_results.append(df)

full_df = pd.concat(all_results)
full_df.to_csv("results/exp3_results.csv", index=False)

# Plot Metric and Entropy
plt.figure(figsize=(15, 10))

for i, name in enumerate([c["name"] for c in exp_configs]):
    sub = full_df[full_df["experiment"] == name]
    
    plt.subplot(2, 2, i+1)
    plt.plot(sub["alpha"], sub["metric"], label="Metric g(alpha)")
    plt.title(f"{name}: Information Metric")
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(2, 2, i+3)
    plt.plot(sub["alpha"], sub["entropy"], label="Entropy", color='orange')
    plt.title(f"{name}: Output Entropy")
    plt.grid(True)

plt.tight_layout()
plt.savefig("results/exp3_analysis.png")
print("Analysis plots saved to results/exp3_analysis.png")

