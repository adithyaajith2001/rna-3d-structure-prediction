"""
train.py — Training loop with early stopping

Key concepts implemented here:
  - Mini-batch gradient descent via DataLoader
  - Adam optimizer with weight decay (L2 regularization)
  - ReduceLROnPlateau scheduler — halves learning rate when progress stalls
  - Early stopping — stops training when validation loss stops improving
  - Best model checkpointing — saves weights at best validation epoch

Why early stopping matters:
  Without it, the model memorizes training data (overfitting).
  Validation loss starts rising while training loss keeps falling.
  We stop at the epoch where validation loss is lowest.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


def train_model(model, X, y, epochs=300, batch_size=128,
                lr=1e-3, patience=25, save_path="best_model.pth",
                device="cuda"):
    """
    Train a coordinate predictor with early stopping.

    Args:
        model      : nn.Module
        X          : numpy array [N, 640] embeddings
        y          : numpy array [N, 3]   coordinates
        epochs     : max training epochs
        batch_size : samples per gradient update
        lr         : initial learning rate
        patience   : early stopping patience (epochs without improvement)
        save_path  : where to save best model weights
        device     : "cuda" or "cpu"

    Returns:
        train_history, val_history : lists of per-epoch RMSD
    """
    # Split data
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t  = torch.FloatTensor(y_tr).to(device)
    X_vl_t  = torch.FloatTensor(X_vl).to(device)
    y_vl_t  = torch.FloatTensor(y_vl).to(device)

    # drop_last=True required by BatchNorm — avoids batch size 1 error
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val   = float("inf")
    patience_c = 0
    tr_hist, vl_hist = [], []

    print(f"{'Epoch':>6} {'Train RMSD':>12} {'Val RMSD':>12}")
    print("-" * 35)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        tr_rmsd = np.sqrt(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            vl_rmsd = np.sqrt(
                criterion(model(X_vl_t), y_vl_t).item())

        scheduler.step(vl_rmsd)
        tr_hist.append(tr_rmsd)
        vl_hist.append(vl_rmsd)

        if vl_rmsd < best_val:
            best_val   = vl_rmsd
            patience_c = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_c += 1

        if (epoch + 1) % 30 == 0:
            print(f"{epoch+1:>6} {tr_rmsd:>12.2f} A {vl_rmsd:>10.2f} A")

        if patience_c >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best val RMSD: {best_val:.2f} A")
    model.load_state_dict(torch.load(save_path))
    return tr_hist, vl_hist
