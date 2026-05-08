"""
features.py — RNA-FM embedding extraction

RNA-FM is a transformer model pretrained on 23 million RNA sequences.
It converts each nucleotide into a 640-dimensional vector encoding:
  - Local sequence context
  - Long-range interactions across the full sequence
  - Evolutionary patterns from millions of related sequences

This is 128x richer than simple integer encoding (A=0, U=1, G=2, C=3).
"""

import torch
import numpy as np
from multimolecule import RnaFmModel, RnaTokenizer


def load_rna_fm(device="cuda"):
    """Load RNA-FM pretrained model from HuggingFace."""
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model = RnaFmModel.from_pretrained("multimolecule/rnafm")
    model = model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"RNA-FM loaded — {params/1e6:.1f}M parameters")
    return model, tokenizer


def get_embeddings(sequence, model, tokenizer, device="cuda"):
    """
    Convert RNA sequence to per-nucleotide 640-dim embeddings.

    Args:
        sequence : str  e.g. "AUGCUAGCUA"
        model    : RNA-FM model
        tokenizer: RNA-FM tokenizer

    Returns:
        numpy array of shape [seq_len, 640]
    """
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Trim special start/end tokens → shape [seq_len, 640]
    return outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()


def build_coordinate_dataset(structures_dict, model, tokenizer, device="cuda"):
    """
    Build (embedding, coordinate) pairs from a dict of PDB structures.

    Args:
        structures_dict : {pdb_id: DataFrame with columns chain,resid,resname,x,y,z}

    Returns:
        X : numpy array [N, 640] — embeddings
        y : numpy array [N, 3]  — centered coordinates
    """
    all_emb, all_coords = [], []

    for pdb_id, struct_df in structures_dict.items():
        for chain_id in struct_df["chain"].unique():
            chain_df = (struct_df[struct_df["chain"] == chain_id]
                        .sort_values("resid")
                        .reset_index(drop=True))
            sequence = "".join(chain_df["resname"].tolist())
            if len(sequence) < 10 or len(sequence) > 500:
                continue
            try:
                emb = get_embeddings(sequence, model, tokenizer, device)
            except Exception:
                continue
            coords = chain_df[["x", "y", "z"]].values
            coords = coords - coords.mean(axis=0)  # center at origin
            min_len = min(len(emb), len(coords))
            for i in range(min_len):
                all_emb.append(emb[i])
                all_coords.append(coords[i])

    return np.array(all_emb), np.array(all_coords)
