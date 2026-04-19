# Interpolation for Incongruent Concepts

This project investigates the geometry of representation spaces in Large Language Models (LLMs) by analyzing the interpolation paths between incongruent (unrelated or mutually exclusive) concepts.

## Key Findings
- **Manifold Curvature**: Interpolation between incongruent concepts results in a **480x higher** information metric (KL-divergence rate) compared to related concepts, indicating severe non-linearity.
- **Off-Manifold Drift**: Incongruent paths show a **30% higher** Sparse Autoencoder (SAE) reconstruction error at the midpoint, suggesting linear interpolation exits the semantic manifold.
- **Uncertainty Spikes**: Output distribution entropy increases significantly during incongruent transitions, whereas related concepts maintain low entropy.
- **Feature Sparsification**: Forced incongruence causes the model to "shut down" specialized features, resulting in a **27-point drop** in L0 sparsity in the SAE feature space.

## Project Structure
- `src/`: Source code for activation extraction, interpolation, and analysis.
- `results/`: CSV data and visualization plots (`analysis.png`, `final_comparison.png`).
- `REPORT.md`: Detailed research report with quantitative findings and discussion.
- `planning.md`: Original research plan and methodology.

## Reproducibility
1.  **Environment**: Use Python 3.10+ and install dependencies:
    ```bash
    uv pip install torch transformer_lens sae_lens matplotlib pandas numpy
    ```
2.  **Run Experiments**:
    ```bash
    python src/experiment_3.py
    ```
3.  **Analyze Results**:
    ```bash
    python src/final_analysis.py
    ```

## References
This work builds on the Linear Representation Hypothesis (LRH) and recent work on Information Geometry and Sparse Autoencoders. See `REPORT.md` for a full list of references.
