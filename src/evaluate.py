"""
evaluate.py — Scoring and visualization

RMSD (Root Mean Square Deviation) is the standard metric for comparing
predicted vs actual 3D structures. Lower is better.

    RMSD = sqrt( mean( (pred_i - actual_i)^2 ) )

Both structures are centered at origin before comparison to remove
positional bias — we care about shape, not absolute position.

RMSD interpretation for RNA:
  < 4 A  = excellent
  4-8 A  = good
  8-15 A = acceptable
  > 15 A = poor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def rmsd_score(pred_df, actual_df):
    """
    Compute RMSD between predicted and actual structures.
    Both are centered at origin before comparison.
    """
    n      = min(len(pred_df), len(actual_df))
    pred   = pred_df[["x","y","z"]].values[:n]
    actual = actual_df[["x","y","z"]].values[:n]
    pred   = pred   - pred.mean(axis=0)
    actual = actual - actual.mean(axis=0)
    return float(np.sqrt(((pred - actual)**2).sum(axis=1).mean()))


def predict_structure(sequence, model, model_rnafm,
                       tokenizer, device="cuda"):
    """Full pipeline: sequence string -> coordinate DataFrame."""
    from src.features import get_embeddings
    emb = get_embeddings(sequence, model_rnafm, tokenizer, device)
    model.eval()
    with torch.no_grad():
        coords = model(
            torch.FloatTensor(emb).to(device)).cpu().numpy()
    return pd.DataFrame({
        "resid"  : range(1, len(coords)+1),
        "resname": list(sequence[:len(coords)]),
        "x": coords[:,0], "y": coords[:,1], "z": coords[:,2]
    })


def plot_model_comparison(structures, titles, colors, save_path=None):
    """
    Plot multiple RNA structures side by side in 3D.

    Args:
        structures : list of DataFrames with x,y,z columns
        titles     : list of strings
        colors     : list of color strings
    """
    fig = plt.figure(figsize=(6*len(structures), 6))
    for idx, (data, title, color) in enumerate(
            zip(structures, titles, colors)):
        ax = fig.add_subplot(1, len(structures), idx+1, projection="3d")
        x  = data["x"] - data["x"].mean()
        y  = data["y"] - data["y"].mean()
        z  = data["z"] - data["z"].mean()
        ax.plot(x, y, z, "o-", color=color, markersize=4, linewidth=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (A)")
        ax.set_ylabel("Y (A)")
        ax.set_zlabel("Z (A)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_curve(train_hist, val_hist, best_epoch, save_path=None):
    """Plot train vs validation RMSD over epochs."""
    plt.figure(figsize=(12, 5))
    plt.plot(train_hist, label="Training RMSD",   color="#3498DB", linewidth=2)
    plt.plot(val_hist,   label="Validation RMSD", color="#E74C3C", linewidth=2)
    plt.axvline(x=best_epoch-1, color="green", linestyle="--",
                label=f"Best model (epoch {best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("RMSD (A)")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
