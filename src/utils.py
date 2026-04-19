import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import numpy as np
import matplotlib.pyplot as plt

def get_activations(model, prompt, layer, pos=-1):
    """Get activations for a specific layer and position."""
    _, cache = model.run_with_cache(prompt)
    # residual stream is usually at 'blocks.L.hook_resid_post'
    return cache[f"blocks.{layer}.hook_resid_post"][0, pos, :]

def interpolate(v1, v2, steps=11):
    """Linear interpolation between two vectors."""
    alphas = np.linspace(0, 1, steps)
    return torch.stack([(1-alpha)*v1 + alpha*v2 for alpha in alphas]), alphas

def get_logits(model, activations):
    """Map activations to logits using the model's unembedding head."""
    # Apply LayerNorm if necessary (GPT-2 uses LN at the end of residual stream)
    # For a specific layer, we might want to apply the remaining layers or just the unembedding.
    # The 'tuned lens' approach uses a trained linear map, but 'logit lens' just uses the final LN + Unembed.
    
    # Simple logit lens:
    resid_normalized = model.ln_final(activations)
    logits = model.unembed(resid_normalized)
    return logits

def get_sae_info(sae, activations):
    """Get SAE features and reconstruction error."""
    # activations: [batch, d_model]
    feature_acts = sae.encode(activations)
    reconstruction = sae.decode(feature_acts)
    mse = torch.mean((activations - reconstruction)**2, dim=-1)
    l0 = (feature_acts > 0).float().sum(dim=-1)
    return feature_acts, mse, l0

