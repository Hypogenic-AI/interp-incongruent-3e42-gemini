from sae_lens import SAE
import torch

try:
    # Just list some releases
    from sae_lens import list_saes
    releases = ["gpt2-small-res-jb"]
    for release in releases:
        print(f"Release: {release}")
        # Not sure if there is a list_sae_ids
        pass
except ImportError:
    print("Cannot import list_saes from sae_lens.")

