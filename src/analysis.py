import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

df = pd.read_csv("results/exp2_results.csv")

experiments = df["experiment"].unique()

plt.figure(figsize=(15, 10))

for i, exp in enumerate(experiments):
    sub = df[df["experiment"] == exp].sort_values("alpha")
    
    alphas = sub["alpha"].values
    p1 = sub["p1"].values
    p2 = sub["p2"].values
    mse = sub["mse"].values
    l0 = sub["l0"].values
    
    # Interpolation Gap: How much probability we lose
    sum_p = p1 + p2
    min_sum_p = np.min(sum_p)
    
    # MSE Peak
    max_mse = np.max(mse)
    
    # L0 Superposition
    mid_l0 = l0[len(l0)//2]
    avg_end_l0 = (l0[0] + l0[-1]) / 2
    l0_diff = mid_l0 - avg_end_l0
    
    print(f"--- {exp} ---")
    print(f"Min Sum Prob: {min_sum_p:.4f}")
    print(f"Max MSE: {max_mse:.4f}")
    print(f"L0 Diff (mid - ends): {l0_diff:.1f}")
    
    # Plot normalized Probabilities
    plt.subplot(2, 2, i+1)
    # We'll normalize by max probability to see the "shape" of the transition
    plt.plot(alphas, p1/np.max(p1), label="P1 (norm)")
    plt.plot(alphas, p2/np.max(p2), label="P2 (norm)")
    plt.plot(alphas, sum_p/(np.max(p1)+np.max(p2)), label="Total (norm)", linestyle='--')
    plt.title(f"{exp}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("results/normalized_probs.png")
print("\nNormalized probability plots saved to results/normalized_probs.png")

# Now compute the "Information Metric" (KL divergence rate)
# We need the full logit distribution for this, but we don't have it in CSV.
# We'll have to redo it in experiment_3.py or just use the P1, P2 as a proxy.
