# Literature Review: Interpolation for Incongruent Concepts

## Research Area Overview
The Linear Representation Hypothesis (LRH) posits that high-level semantic concepts are represented as directions (vectors) in the activation space of neural networks. Recent work has extended this to explore the geometry of these representations (information geometry) and how they handle multiple concepts simultaneously (superposition). A key area of interest is how models behave when we interpolate between "incongruent" concepts—concepts that are not naturally related or causally separable.

## Key Papers

### 1. Schrödinger's Bat: Diffusion Models Sometimes Generate Polysemous Words in Superposition
- **Authors**: Jennifer C. White, Ryan Cotterell
- **Year**: 2022
- **Source**: arXiv (2211.13095)
- **Key Contribution**: Demonstrates that diffusion models can generate images containing multiple senses of a polysemous word because CLIP encodes these senses in linear superposition.
- **Methodology**: Used weighted sums of CLIP encodings ($\alpha_1 \text{CLIP}(s_1) + \alpha_2 \text{CLIP}(s_2)$) to steer Stable Diffusion.
- **Relevance**: Directly shows that "interpolation" (weighted sum) of incongruent concepts (different word senses) leads to a visual superposition in diffusion models.

### 2. Toy Models of Superposition
- **Authors**: Nelson Elhage et al. (Anthropic)
- **Year**: 2022
- **Source**: arXiv (2209.10652)
- **Key Contribution**: Provides a theoretical and empirical framework for understanding how models store more features than they have dimensions by using superposition.
- **Methodology**: Trained small ReLU networks on synthetic sparse features.
- **Key Findings**: Features are organized into geometric structures (polytopes) to minimize interference.
- **Relevance**: Explains the underlying mechanism (superposition) that allows incongruent concepts to coexist in the same representation space.

### 3. The Linear Representation Hypothesis and the Geometry of Large Language Models
- **Authors**: Kiho Park et al.
- **Year**: 2023
- **Source**: arXiv (2311.03658)
- **Key Contribution**: Investigates the LRH from a geometric perspective, finding that representation spaces often have non-Euclidean structures.
- **Relevance**: Suggests that simple linear interpolation might not be the "natural" path between concepts in LLMs.

### 4. On the Origins of Linear Representations in Large Language Models
- **Authors**: Yibo Jiang et al.
- **Year**: 2024
- **Source**: arXiv (2403.03867)
- **Key Contribution**: Shows that independent concepts are represented almost orthogonally due to the next-token prediction objective and gradient descent bias.
- **Relevance**: Orthogonality between concepts is a prerequisite for clean interpolation but also leads to "incongruent" mixtures when summed.

### 5. Sparse Autoencoders Find Highly Interpretable Features in Language Models
- **Authors**: Cunningham et al.
- **Year**: 2023
- **Source**: arXiv (2309.08600)
- **Key Contribution**: Uses Sparse Autoencoders (SAEs) to decompose polysemantic neurons into monosemantic features.
- **Relevance**: Provides tools (SAEs) to isolate and then interpolate between specific concepts without the "noise" of other features in the same neuron.

## Common Methodologies
- **Steering/Intervention**: Adding or subtracting concept vectors (e.g., from CLIP or LLM activations) to change model behavior.
- **Activation Probing**: Training linear classifiers to detect concepts in representation space.
- **Synthetic "Toy" Models**: Using small, controlled models to study fundamental properties like superposition.
- **Information Geometry**: Analyzing the metric structure of the representation space (e.g., using Fisher Information Metric).

## Standard Baselines
- **Linear Interpolation (LERP)**: $v = (1-t)v_1 + t v_2$.
- **Spherical Linear Interpolation (SLERP)**: Used for normalized vectors (common in GANs/Diffusion).
- **Concept Activation Vectors (CAV)**: Standard method for identifying concept directions.

## Evaluation Metrics
- **Logit Lens / Tuned Lens**: Measuring how interpolation affects output probabilities.
- **Visual Inspection**: In generative models, checking if both concepts appear (as in Schrödinger's Bat).
- **SAE Reconstruction Error**: Checking if the interpolated point remains "on-manifold" according to a Sparse Autoencoder.

## Datasets in the Literature
- **WiC (Word in Context)**: For studying polysemy.
- **Geometry of Truth (Cities, Facts)**: For studying simple, binary semantic concepts.
- **MS-COCO / LAION**: For large-scale image-text concept studies.
- **Synthetic Sparse Features**: As used in Toy Models of Superposition.

## Recommendations for Our Experiment
1. **Synthetic Setup**: Replicate the Toy Models of Superposition approach to study how the model handles "incongruent" (independent) features vs. "congruent" (correlated) features during interpolation.
2. **LLM Probing**: Use `TransformerLens` to extract activations for incongruent pairs (e.g., "Paris" and "Justice") and observe the "in-between" states.
3. **SAE Validation**: Use `SAELens` to see if interpolated vectors are recognized as valid combinations of features or if they fall into "empty" space.
4. **Diffusion Steering**: Replicate the "Schrödinger's Bat" weighted sum experiment with more varied incongruent concepts to see the limits of superposition.
