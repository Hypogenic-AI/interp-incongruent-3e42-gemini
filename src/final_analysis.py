import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results/exp3_results.csv")

def analyze_exp(name):
    sub = df[df["experiment"] == name].sort_values("alpha")
    
    # Max Metric (excluding last point)
    max_metric = sub["metric"].iloc[:-1].max()
    # Mean Metric
    mean_metric = sub["metric"].iloc[:-1].mean()
    # Max MSE
    max_mse = sub["mse"].max()
    # MSE at alpha=0.5
    mid_mse = sub[sub["alpha"] == 0.5]["mse"].values[0]
    
    # Entropy trend
    start_ent = sub["entropy"].iloc[0]
    mid_ent = sub[sub["alpha"] == 0.5]["entropy"].values[0]
    end_ent = sub["entropy"].iloc[-1]
    
    return {
        "max_metric": max_metric,
        "mean_metric": mean_metric,
        "max_mse": max_mse,
        "mid_mse": mid_mse,
        "entropy_peak": mid_ent - (start_ent + end_ent)/2
    }

related = analyze_exp("Related")
incongruent = analyze_exp("Incongruent")

print("--- Comparison ---")
print(f"{'Metric':<15} | {'Related':<10} | {'Incongruent':<10}")
print("-" * 45)
for k in related.keys():
    print(f"{k:<15} | {related[k]:<10.4f} | {incongruent[k]:<10.4f}")

# Visualization of Metric vs MSE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for name in ["Related", "Incongruent"]:
    sub = df[df["experiment"] == name]
    plt.plot(sub["alpha"], sub["metric"], label=name)
plt.title("Information Metric (log scale)")
plt.yscale('log')
plt.legend()

plt.subplot(1, 2, 2)
for name in ["Related", "Incongruent"]:
    sub = df[df["experiment"] == name]
    plt.plot(sub["alpha"], sub["mse"], label=name)
plt.title("SAE Reconstruction MSE")
plt.legend()

plt.tight_layout()
plt.savefig("results/final_comparison.png")

