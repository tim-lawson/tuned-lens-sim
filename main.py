from os import PathLike
import sqlite3
import torch
from sae_lens import SAE
from tqdm import tqdm
from tuned_lens import TunedLens
from tuned_lens.nn.unembed import Unembed
from transformer_lens import HookedTransformer


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def create_db(database: str | PathLike) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cosines (
        layer_i INTEGER,
        layer_j INTEGER,
        latent_i INTEGER,
        latent_j INTEGER,
        cosine REAL,
        PRIMARY KEY (layer_i, layer_j, latent_i, latent_j)
    )
    """)
    conn.commit()
    return conn, cursor


def batch_insert(cursor: sqlite3.Cursor, data: list[tuple]) -> None:
    cursor.executemany("INSERT OR REPLACE INTO cosines VALUES (?, ?, ?, ?, ?)", data)


def normalize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / torch.max(norm, 1e-8 * torch.ones_like(norm))


if __name__ == "__main__":
    device = get_device()

    model = HookedTransformer.from_pretrained("gpt2", device=device, fold_ln=False)
    model = model.requires_grad_(False)

    tuned_lens = TunedLens.from_unembed_and_pretrained(Unembed(model), "gpt2")
    tuned_lens = tuned_lens.requires_grad_(False)
    tuned_lens = tuned_lens.to(device)

    saes = [
        SAE.from_pretrained(
            release="gpt2-small-resid-post-v5-32k",
            sae_id=f"blocks.{layer}.hook_resid_post",
            device=str(device),
        )[0].requires_grad_(False)
        for layer in range(12)
    ]

    W_decs = [
        tuned_lens.transform_hidden(sae.W_dec, idx) for idx, sae in enumerate(saes)
    ]

    del model, tuned_lens, saes

    conn, cursor = create_db("cosines.db")
    batch_size = 100_000

    for layer_i, W_dec_i in tqdm(enumerate(W_decs)):
        for layer_j, W_dec_j in tqdm(enumerate(W_decs)):
            W_dec_i, W_dec_j = normalize(W_dec_i), normalize(W_dec_j)
            cosines = torch.mm(W_dec_i, W_dec_j.T)

            batch = []
            for i in range(cosines.shape[0]):
                for j in range(cosines.shape[1]):
                    batch.append((layer_i, layer_j, i, j, cosines[i, j].item()))
                    if len(batch) >= batch_size:
                        batch_insert(cursor, batch)
                        batch = []
            if batch:
                batch_insert(cursor, batch)
        conn.commit()
    conn.close()
