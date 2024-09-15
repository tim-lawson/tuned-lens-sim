from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import pandas as pd
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
from tuned_lens import TunedLens
from tuned_lens.nn.unembed import Unembed


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def normalize(x: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / torch.max(norm, eps * torch.ones_like(norm))


def mmcs(W_decs: Sequence[torch.Tensor], prefix: str) -> None:
    mcs, mmcs = [], []

    for layer1, layer2 in pairwise(range(len(W_decs))):
        W_dec1 = normalize(W_decs[layer1], dim=-1)
        W_dec2 = normalize(W_decs[layer2], dim=-1)
        values, indices = torch.max(torch.mm(W_dec1, W_dec2.T), dim=1)
        mcs += [
            (layer1, layer2, latent1, latent2.item(), value.item())
            for latent1, latent2, value in zip(
                range(len(indices)), indices, values, strict=True
            )
        ]
        mmcs.append((layer1, layer2, values.mean().item()))

    pd.DataFrame(mcs, columns=["layer1", "layer2", "latent1", "latent2", "mcs"]).to_csv(
        f"{prefix}_mcs.csv", index=False
    )
    pd.DataFrame(mmcs, columns=["layer1", "layer2", "mmcs"]).to_csv(
        f"{prefix}_mmcs.csv", index=False
    )


def main(model_name: str, release: str, sae_id: Callable[[int], str]) -> None:
    device = get_device()

    model = HookedTransformer.from_pretrained(model_name, device=device, fold_ln=False)
    model = model.requires_grad_(False)

    lens = TunedLens.from_unembed_and_pretrained(
        Unembed(model), model_name, map_location=device
    )
    lens = lens.requires_grad_(False)
    lens = lens.to(device)

    saes = [
        SAE.from_pretrained(
            release=release,
            sae_id=sae_id(layer),
            device=str(device),
        )[0].requires_grad_(False)
        for layer in range(model.cfg.n_layers)
    ]

    mmcs([sae.W_dec for sae in saes], f"{release}_base")
    mmcs(
        [lens.transform_hidden(sae.W_dec, layer) for layer, sae in enumerate(saes)],
        f"{release}_lens",
    )


@dataclass
class Config:
    model_name: str
    release: str
    sae_id: Callable[[int], str]


if __name__ == "__main__":
    for config in [
        Config(
            "EleutherAI/pythia-70m-deduped",
            "pythia-70m-deduped-res-sm",
            lambda layer: f"blocks.{layer}.hook_resid_post",
        ),
        Config(
            "EleutherAI/pythia-70m-deduped",
            "pythia-70m-deduped-mlp-sm",
            lambda layer: f"blocks.{layer}.hook_mlp_out",
        ),
        Config(
            "EleutherAI/pythia-70m-deduped",
            "pythia-70m-deduped-att-sm",
            lambda layer: f"blocks.{layer}.hook_attn_out",
        ),
        Config(
            "gpt2",
            "gpt2-small-resid-post-v5-32k",
            lambda layer: f"blocks.{layer}.hook_resid_post",
        ),
        Config(
            "gpt2",
            "gpt2-small-res-jb",
            lambda layer: f"blocks.{layer}.hook_resid_pre",
        ),
        Config(
            "gpt2",
            "gpt2-small-resid-post-v5-32k",
            lambda layer: f"blocks.{layer}.hook_resid_post",
        ),
        Config(
            "gpt2",
            "gpt2-small-resid-mid-v5-32k",
            lambda layer: f"blocks.{layer}.hook_resid_mid",
        ),
        Config(
            "gpt2",
            "gpt2-small-mlp-out-v5-32k",
            lambda layer: f"blocks.{layer}.hook_mlp_out",
        ),
        Config(
            "gpt2",
            "gpt2-small-attn-out-v5-32k",
            lambda layer: f"blocks.{layer}.hook_attn_out",
        ),
    ]:
        main(config.model_name, config.release, config.sae_id)
