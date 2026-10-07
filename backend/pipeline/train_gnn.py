"""
Training Graph Neural Network (GNN) on Molecular Graphs for hERG Risk Prediction.
Evaluated on the exact same data splits as the classical model.
Saves PyTorch weights to artifacts/gnn_model.pt.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.models.gnn import MolecularGNN, mol_to_graph_data, collate_graph_batch, NODE_FEATURE_DIM

PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class MolecularDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.graphs = []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is not None:
                g = mol_to_graph_data(mol)
                if g is not None:
                    g["y"] = float(row["y"])
                    self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def main():
    print("=" * 60)
    print("TRAINING GRAPH NEURAL NETWORK (GNN) BENCHMARK")
    print("=" * 60)

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    print(f"Constructing molecular graphs for Train ({len(train_df)}) and Val ({len(val_df)})...")
    t0 = time.time()
    train_dataset = MolecularDataset(train_df)
    val_dataset = MolecularDataset(val_df)
    print(f"Graphs created in {round(time.time() - t0, 2)}s.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_graph_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_graph_batch
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = MolecularGNN(in_dim=NODE_FEATURE_DIM, hidden_dim=96, num_layers=3, dropout=0.20)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    num_epochs = 12
    best_val_auc = 0.0
    best_state = None

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        t_epoch_start = time.time()

        for batch in train_loader:
            if not batch or "y" not in batch:
                continue

            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            batch_idx = batch["batch"].to(device)
            y = batch["y"].to(device)
            num_graphs = batch["num_graphs"]

            optimizer.zero_grad()
            logits = model(x, edge_index, batch_idx, num_graphs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * num_graphs

        avg_train_loss = total_loss / len(train_dataset)

        # Validation evaluation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if not batch or "y" not in batch:
                    continue
                x = batch["x"].to(device)
                edge_index = batch["edge_index"].to(device)
                batch_idx = batch["batch"].to(device)
                num_graphs = batch["num_graphs"]

                logits = model(x, edge_index, batch_idx, num_graphs)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(batch["y"].numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        scheduler.step(val_auc)
        epoch_dur = round(time.time() - t_epoch_start, 1)

        print(f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_dur}s] - Train Loss: {avg_train_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()

    # Save best model
    if best_state is not None:
        model_path = ARTIFACTS_DIR / "gnn_model.pt"
        torch.save(best_state, model_path)
        print(f"\n[SUCCESS] Best GNN checkpoint (Val ROC-AUC: {best_val_auc:.4f}) saved to {model_path}")


if __name__ == "__main__":
    main()
