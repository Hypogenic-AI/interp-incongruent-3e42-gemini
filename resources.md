# Resources Catalog: Interpolation for Incongruent Concepts

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 8

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Schrödinger's Bat | White & Cotterell | 2022 | papers/2211.13095_Schrodingers_Bat.pdf | Polysemy & Superposition in Diffusion |
| Toy Models of Superposition | Elhage et al. | 2022 | papers/2209.10652_Toy_Models_of_Superposition.pdf | Theoretical framework for superposition |
| Sparse Autoencoders | Cunningham et al. | 2023 | papers/2309.08600_Sparse_Autoencoders.pdf | Feature decomposition |
| Linear Representation Hypothesis | Park et al. | 2023 | papers/2311.03658_Linear_Representation_Hypothesis_Geometry.pdf | Geometry of representation space |
| Geometry of Truth | Marks & Tegmark | 2023 | papers/2310.06824_Geometry_of_Truth.pdf | Linear structure of truth/falsehood |
| Language Models Space/Time | Gurnee & Tegmark | 2023 | papers/2310.02207_Space_and_Time.pdf | Spatial/temporal representations |
| Origins of Linear Repr. | Jiang et al. | 2024 | papers/2403.03867_Origins_of_Linear_Representations.pdf | Theoretical origins of LRH |
| Info. Geometry of Softmax | Park et al. | 2026 | papers/2602.15293_Information_Geometry_Softmax.pdf | Latest work on geometric probing |

## Datasets

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Cities | Geometry of Truth | ~1000 rows | Truthfulness | datasets/cities.csv | Geography concepts |
| Common Claim | Geometry of Truth | ~20000 rows | Truthfulness | datasets/common_claim.csv | General knowledge |
| SP-EN Trans | Geometry of Truth | ~1000 rows | Translation | datasets/sp_en_trans.csv | Translation concepts |
| WiC Sample | HuggingFace | 100 rows | Polysemy | datasets/wic_sample/ | Word in Context (sample) |

## Code Repositories

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Toy Models | github.com/anthropics/toy-models-of-superposition | Synthetic experiments | code/toy-models/ | Essential for phase changes |
| Geometry of Truth | github.com/saprmarks/geometry-of-truth | Concept probing | code/geometry-of-truth/ | Data & probes |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic Interpretability | code/transformer-lens/ | Activation extraction |
| SAELens | github.com/jbloomAus/SAELens | Sparse Autoencoders | code/saelens/ | Feature analysis |
| Schrödinger's Bat | github.com/rycolab/diffusion-polysemy | Diffusion steering | code/schrodingers-bat/ | Polysemy experiments |

## Resource Gathering Notes

### Search Strategy
- Used `find_papers.py` for initial literature search.
- Verified arXiv IDs manually to ensure correct papers were downloaded.
- Cloned official repositories associated with the most relevant papers.
- Gathered datasets specifically mentioned in or used by the papers.

### Challenges Encountered
- Some initial arXiv IDs were incorrect (e.g., 2211.05100 was BLOOM, not Schrödinger's Bat). Fixed by searching by title.
- Many datasets are large; only samples or small CSV files were downloaded to keep the workspace manageable.

## Recommendations for Experiment Design

1. **Primary Dataset**: Use the `cities.csv` and `common_claim.csv` for initial conceptual interpolation experiments in LLMs.
2. **Baseline Methods**: Use LERP between concept activation vectors (CAV) as the primary baseline.
3. **Advanced Method**: Use Sparse Autoencoder (SAELens) to find feature directions and interpolate between "incongruent" features to see if the model produces "blended" or "superposed" outputs.
4. **Validation**: Use the Logit Lens (via TransformerLens) to observe how output distributions change during interpolation.
