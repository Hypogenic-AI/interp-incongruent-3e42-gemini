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

# Experiment 2: Contextual Interpolation
exp_configs = [
    {
        "name": "Related (France-England Capitals)",
        "prompt1": "The capital of France is",
        "prompt2": "The capital of England is",
        "target1": " Paris",
        "target2": " London"
    },
    {
        "name": "Incongruent (France Capital - Cat Food)",
        "prompt1": "The capital of France is",
        "prompt2": "The favorite food of a cat is",
        "target1": " Paris",
        "target2": " fish"
    },
    {
        "name": "Opposite (Truth-Falsehood)",
        "prompt1": "The city of Paris is in France.",
        "prompt2": "The city of Paris is in England.",
        "target1": " True", # Note: GPT-2 might not use " True"/" False" directly
        "target2": " False"
    }
]

# We'll use specific tokens for the logit lens
# For Truth-Falsehood, we'll look at " True" vs " False" or similar
# Actually, let's just stick to the targets provided.

all_results = []

for config in exp_configs:
    name = config["name"]
    p1 = config["prompt1"]
    p2 = config["prompt2"]
    t1 = config["target1"]
    t2 = config["target2"]
    
    print(f"\nProcessing {name}...")
    
    v1 = get_act(p1)
    v2 = get_act(p2)
    
    steps = 21
    v_interp, alphas = interpolate(v1, v2, steps=steps)
    v_interp = v_interp.to(device)
    
    logits = get_logits(model, v_interp)
    probs = F.softmax(logits, dim=-1)
    
    idx1 = model.to_single_token(t1)
    idx2 = model.to_single_token(t2)
    
    prob1 = probs[:, idx1].detach().cpu().numpy()
    prob2 = probs[:, idx2].detach().cpu().numpy()
    
    if sae:
        _, mse, l0 = get_sae_info(sae, v_interp)
        mse = mse.detach().cpu().numpy()
        l0 = l0.detach().cpu().numpy()
    else:
        mse = np.zeros(steps)
        l0 = np.zeros(steps)
        
    df = pd.DataFrame({
        "alpha": alphas,
        "p1": prob1,
        "p2": prob2,
        "mse": mse,
        "l0": l0,
        "experiment": name
    })
    all_results.append(df)

full_df = pd.concat(all_results)
full_df.to_csv("results/exp2_results.csv", index=False)

# Plotting
plt.figure(figsize=(15, 12))

for i, config in enumerate(exp_configs):
    name = config["name"]
    df = full_df[full_df["experiment"] == name]
    
    # 1. Probs
    plt.subplot(3, 3, i*3 + 1)
    plt.plot(df["alpha"], df["p1"], label=config["target1"])
    plt.plot(df["alpha"], df["p2"], label=config["target2"])
    plt.title(f"{name}\nProbabilities")
    plt.legend()
    
    # 2. MSE
    plt.subplot(3, 3, i*3 + 2)
    plt.plot(df["alpha"], df["mse"], color='red')
    plt.title("SAE MSE")
    
    # 3. L0
    plt.subplot(3, 3, i*3 + 3)
    plt.plot(df["alpha"], df["l0"], color='green')
    plt.title("SAE L0")

plt.tight_layout()
plt.savefig("results/exp2_analysis.png")
print("Analysis plots saved to results/exp2_analysis.png")

