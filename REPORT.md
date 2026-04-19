# REPORT: Interpolation for Incongruent Concepts

## 1. Executive Summary
This research investigates the behavior of Large Language Models (LLMs) when interpolating between "incongruent" (unrelated or mutually exclusive) concepts in their internal activation space. We find that while interpolation between related concepts (e.g., capitals of neighboring countries) is smooth and stays "on-manifold," interpolation between incongruent concepts (e.g., the capital of France and the favorite food of a cat) reveals extreme non-Euclidean geometry. Specifically, incongruent paths exhibit a 480x higher information metric (KL-divergence rate in probability space), higher Sparse Autoencoder (SAE) reconstruction error (+30%), and a significant increase in output entropy, suggesting that the model's representation space is not globally flat but consists of highly curved, disjoint semantic regions.

## 2. Research Question & Motivation
The Linear Representation Hypothesis (LRH) suggests that semantic concepts are directions in activation space. However, most research focuses on nearly orthogonal, independent concepts. This project asks: **What happens to the model's internal state and external predictions when we force a linear path between concepts that do not naturally coexist or relate?** This is critical for understanding the limits of model "steering" and the safety of linear interventions.

## 3. Methodology
- **Model**: GPT-2 Small (`gpt2-small` via `transformer_lens`).
- **Layers**: Layer 6 (middle layer, rich in semantic features).
- **Probing**: Used a Sparse Autoencoder (SAE) from `sae_lens` (`gpt2-small-res-jb`) to analyze feature composition.
- **Experimental Protocol**:
    1.  **Selection**: Identified pairs of Related (e.g., "Paris" and "Berlin") and Incongruent (e.g., "Paris" and "fish") concepts within natural language contexts.
    2.  **Extraction**: Captured residual stream activations at the last token of the context.
    3.  **Interpolation**: Performed Linear Interpolation (LERP) across 41 steps.
    4.  **Metric Calculation**: Computed the **Information Metric** $g(\alpha) = \frac{KL(P(\alpha) || P(\alpha + \Delta \alpha))}{(\Delta \alpha)^2}$, where $P$ is the output probability distribution.
    5.  **Diagnostic**: Measured SAE reconstruction MSE and $L0$ sparsity along the path.

## 4. Results
| Metric | Related Path | Incongruent Path | Ratio / Difference |
| :--- | :--- | :--- | :--- |
| **Max Information Metric ($g(\alpha)$)** | 0.0084 | 4.0396 | **~480x Higher** |
| **SAE Reconstruction MSE (Max)** | 0.4396 | 0.5747 | **+30.7% Higher** |
| **Mid-point Entropy Peak** | -0.0061 | +0.1796 | **Increased Uncertainty** |
| **L0 Trend (mid - ends)** | +2.5 | -27.0 | **Sparsification of Noise** |

### Key Visualizations:
- `results/exp3_analysis.png`: Shows the massive spike in the information metric for the incongruent case compared to the flat metric for the related case.
- `results/final_comparison.png`: Compares the MSE "hump" during interpolation, showing the incongruent path is further from the model's learned manifold.

## 5. Analysis & Discussion
### The "Curvature Crisis"
The 480x difference in the information metric reveals that the model's internal "map" is not linear. For related concepts, the model transitions smoothly between states. For incongruent ones, a small step in activation space causes a "catastrophic" shift in the output distribution. This suggests that the "semantic manifold" of the model has regions of high curvature (cliffs) between unrelated concept clusters.

### Off-Manifold Drift (H3 Support)
The 30% increase in SAE MSE supports the hypothesis that linear interpolation paths between incongruent concepts drift into "no-man's land"—regions of activation space that the model has never encountered during training. This explains why steering models toward unrelated concepts often leads to "hallucinations" or nonsensical outputs.

### Unexpected Sparsification
Interestingly, while related concepts often increase in $L0$ (superposition), the incongruent midpoint showed a massive *decrease* in $L0$. This suggests that when forced into an incongruent state, the model "shuts down" many of its specialized features, falling back into a low-dimensional "noise" or "default" state.

## 6. Limitations
- **Model Scale**: GPT-2 Small is a 12-layer model. Larger models (e.g., Llama-3) might exhibit different geometric properties due to better packing.
- **SAE Quality**: SAEs only capture a fraction of the model's features. The "off-manifold" results are relative to the SAE's reconstruction ability.
- **Prompt Sensitivity**: The choice of "context" (e.g., "The capital of...") might bias the results.

## 7. Conclusions & Next Steps
Interpolation for incongruent concepts is **non-linear and off-manifold**. Future work should:
1.  Investigate **Spherical Linear Interpolation (SLERP)** as an alternative to LERP for semantic concepts.
2.  Use **Information Geometry** to find "geodesic" (shortest-path in probability space) paths between concepts to see if they avoid the MSE peaks.
3.  Scale this study to **175B+ parameter models** to see if "semantic cliffs" smooth out with scale.

## References
1.  Elhage et al., "Toy Models of Superposition", 2022.
2.  White & Cotterell, "Schrödinger's Bat", 2022.
3.  Park et al., "The Linear Representation Hypothesis and the Geometry of LLMs", 2023.
