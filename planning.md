# Research Plan: Interpolation for Incongruent Concepts

## Motivation & Novelty Assessment

### Why This Research Matters
Linear Representation Hypothesis (LRH) and steering techniques (like CAVs) assume that we can linearly manipulate concepts in activation space. However, most real-world concepts are not perfectly independent. Interpolation between incongruent (mutually exclusive or unrelated) concepts is a "stress test" for LRH. If interpolation leads to "hallucinated" or nonsensical intermediate states, it reveals the limits of linear steering and the underlying geometry of the model's "world model."

### Gap in Existing Work
Existing work (e.g., Toy Models of Superposition) focuses on how features are packed. "Schrödinger's Bat" showed visual superposition in diffusion. However, there is a lack of systematic study on the *information geometry* (the relationship between activation interpolation and probability manifold movement) for *incongruent semantic concepts* in LLMs. We don't fully understand if the model "perceives" the path between "Paris" and "London" as a smooth blend or a leap over a chasm of low probability.

### Our Novel Contribution
We will:
1.  Define "Incongruence" types: **Mutually Exclusive** (Paris vs. London as capitals) and **Unrelated** (Paris vs. Justice).
2.  Map the **Logit Landscape**: Track the output probability distribution along the interpolation path.
3.  **SAE Diagnostic**: Use Sparse Autoencoders to determine if intermediate states are recognized as valid feature combinations or as "interference noise."

### Experiment Justification
- **Experiment 1: Logit Path Analysis**: Essential to see how the model's "belief" shifts. Does it show a "bistable" jump or a "superposed" state?
- **Experiment 2: SAE Feature Decomposition**: Essential to determine if the linear path stays on the "semantic manifold" or if it activates "junk" features.
- **Experiment 3: Information Geometry (Metric Calculation)**: Measure the "distance" in probability space relative to activation space to identify regions of high curvature.

## Research Question
How does the output distribution and internal feature representation of an LLM behave during interpolation between incongruent concepts, and does this behavior reveal non-Euclidean geometry in the representation space?

## Background and Motivation
LRH suggests semantic concepts are vectors. Interpolation is the simplest operation between them. For "congruent" (correlated) concepts, this is well-studied. For "incongruent" ones, it's a frontier. This connects to "superposition" (how models store competing features) and "interference" (when those features clash).

## Hypothesis Decomposition
- **H1 (Bistability)**: For mutually exclusive concepts, the model's output will jump sharply between states rather than showing a smooth blend (Winner-Take-All).
- **H2 (Superposition)**: For unrelated concepts, the model will represent a superposition where both concepts are simultaneously "active" without interference.
- **H3 (Off-Manifold)**: Linear interpolation paths between distant incongruent concepts will pass through "empty" regions of activation space where SAE reconstruction error is high.

## Proposed Methodology

### Approach
We will use `TransformerLens` to extract activations from a mid-sized LLM (e.g., GPT-2 Small or Llama-3-8B if resources permit, but GPT-2 is easier for SAEs) and `SAELens` to analyze the features.

### Experimental Steps
1.  **Selection of Concept Pairs**:
    *   *Congruent*: "France" -> "Spain" (Geographically related).
    *   *Incongruent (Exclusive)*: "The capital of France is [Paris]" vs. "The capital of France is [London]".
    *   *Incongruent (Unrelated)*: "Paris" vs. "Justice".
2.  **Activation Extraction**: Generate activations for prompts representing these concepts.
3.  **Interpolation**: Perform LERP between the mean activation vectors.
4.  **Logit Lens**: Pass interpolated activations through the model's unembedding head to see the output distribution.
5.  **SAE Probing**: Pass interpolated activations through a pre-trained SAE to see which features activate and measure reconstruction error.
6.  **Curvature Measurement**: Calculate the KL-divergence between adjacent steps in the interpolation to find the "information metric."

### Baselines
- **Random Path**: Interpolation between a concept and a random vector.
- **Congruent Path**: Interpolation between related concepts (e.g., "Paris" and "Lyon").

### Evaluation Metrics
- **Entropy of Output Distribution**: High entropy indicates "blending" or "confusion."
- **SAE L0 (Sparsity)**: How many features are active.
- **SAE Reconstruction MSE**: Indicates if the vector is "on-manifold."
- **KL Divergence Rate**: Measuring the "speed" of change in probability space.

### Statistical Analysis Plan
We will perform multiple runs across different layers and different concept pairs to ensure the findings are not specific to one instance. We will use t-tests to compare the average SAE error on congruent vs. incongruent paths.

## Expected Outcomes
- Mutually exclusive concepts will show a "phase transition" (sharp jump) in probability.
- Unrelated concepts will show a "flat" superposition (both features active).
- Incongruent paths will show higher SAE reconstruction error in the middle compared to congruent ones.

## Timeline and Milestones
1.  **Environment Setup**: 15 min.
2.  **Implementation (Activation & SAE)**: 60 min.
3.  **Data Collection & Experiments**: 90 min.
4.  **Analysis & Visualization**: 45 min.
5.  **Documentation**: 30 min.

## Potential Challenges
- Finding good "incongruent" pairs that the model actually understands.
- SAEs might not be available for the layer we want to study.
- GPT-2 might be too simple to show complex semantic incongruence.

## Success Criteria
- Successful visualization of the logit landscape for at least 3 pairs.
- Quantifiable difference in SAE behavior between congruent and incongruent paths.
- A "Metric" map showing where the model "perceives" the largest semantic distance.
