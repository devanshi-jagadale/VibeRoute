"""
stage7_classifier.py — Stage 7: MLP Transition Classifier

Architecture:
  MLP: (131+2k) → 256 → 128 → 64 → 1
  activation:     ReLU (hidden), Sigmoid (output)
  loss:           binary cross-entropy
  optimizer:      Adam
  regularization: dropout 0.3

Usage:
    python stage7_classifier.py              # train from scratch
    python stage7_classifier.py --eval       # evaluate saved model on test set
    python stage7_classifier.py --status     # show model info and exit
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR   = Path("data")
PAIRS_PATH = DATA_DIR / "pairs.npz"
MODEL_PATH = DATA_DIR / "mlp_classifier.pt"
STATS_PATH = DATA_DIR / "classifier_stats.json"

# ── Training config ───────────────────────────────────────────────────────────
BATCH_SIZE   = 512
EPOCHS       = 50
LR           = 1e-3
DROPOUT      = 0.3
PATIENCE     = 5        # early stopping patience
HIDDEN_DIMS  = [256, 128, 64]
TRAIN_SPLIT  = 0.8
VAL_SPLIT    = 0.1
# TEST_SPLIT  = 0.1    (remainder)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_pairs() -> tuple[np.ndarray, np.ndarray]:
    print("📂 Loading pairs...")
    data = np.load(PAIRS_PATH)
    X, y = data["X"], data["y"]
    print(f"   X: {X.shape}  y: {y.shape}")
    print(f"   Positives: {int(y.sum())}  Negatives: {int((1-y).sum())}")
    return X, y


def split_data(X, y):
    n     = len(X)
    n_tr  = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    idx = np.random.permutation(n)
    tr_idx  = idx[:n_tr]
    val_idx = idx[n_tr:n_tr + n_val]
    te_idx  = idx[n_tr + n_val:]

    return (
        X[tr_idx], y[tr_idx],
        X[val_idx], y[val_idx],
        X[te_idx],  y[te_idx],
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(input_dim: int):
    import torch
    import torch.nn as nn

    layers = []
    prev = input_dim
    for h in HIDDEN_DIMS:
        layers += [
            nn.Linear(prev, h),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        ]
        prev = h
    layers.append(nn.Linear(prev, 1))
    layers.append(nn.Sigmoid())

    model = nn.Sequential(*layers)
    print(f"\n🧠 Model architecture:")
    print(f"   Input  : {input_dim}")
    for h in HIDDEN_DIMS:
        print(f"   Hidden : {h}  + ReLU + Dropout({DROPOUT})")
    print(f"   Output : 1  + Sigmoid")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Params : {total_params:,}")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_tr, y_tr, X_val, y_val, input_dim: int):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on: {device}")

    model = build_model(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(device)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    train_ds     = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    patience_count = 0
    best_state = None
    history = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val AUC':>9}  {'Val Acc':>9}")
    print("─" * 55)

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_pred  = model(X_val_t).cpu().numpy().flatten()
            val_loss  = criterion(
                torch.tensor(val_pred).unsqueeze(1),
                y_val_t.cpu()
            ).item()
            val_acc   = ((val_pred > 0.5).astype(float) == y_val).mean()

            # AUC (manual — no sklearn needed in training loop)
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(y_val, val_pred)

        tr_loss = np.mean(train_losses)
        history.append({
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_loss": float(val_loss),
            "val_auc": float(val_auc),
            "val_acc": float(val_acc),
        })

        print(f"{epoch:>6}  {tr_loss:>11.4f}  {val_loss:>10.4f}  {val_auc:>9.4f}  {val_acc:>9.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n  ⏹️  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # restore best weights
    model.load_state_dict(best_state)
    return model, history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_te, y_te, device=None):
    import torch
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        X_t    = torch.tensor(X_te, dtype=torch.float32).to(device)
        preds  = model(X_t).cpu().numpy().flatten()

    labels = (preds > 0.5).astype(int)

    auc  = roc_auc_score(y_te, preds)
    acc  = (labels == y_te).mean()
    prec = precision_score(y_te, labels, zero_division=0)
    rec  = recall_score(y_te, labels, zero_division=0)
    f1   = f1_score(y_te, labels, zero_division=0)
    cm   = confusion_matrix(y_te, labels)

    print(f"\n📊 Test set evaluation:")
    print(f"   AUC       : {auc:.4f}")
    print(f"   Accuracy  : {acc:.4f}")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1        : {f1:.4f}")
    print(f"   Confusion matrix:")
    print(f"     TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"     FN={cm[1,0]}  TP={cm[1,1]}")

    return {"auc": auc, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, input_dim: int, stats: dict, history: list):
    import torch
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  input_dim,
        "hidden_dims": HIDDEN_DIMS,
        "dropout":    DROPOUT,
    }, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    with open(STATS_PATH, "w") as f:
        json.dump({"stats": stats, "history": history, "input_dim": input_dim}, f, indent=2)
    print(f"📄 Stats saved → {STATS_PATH}")


def load_model():
    import torch
    import torch.nn as nn

    ckpt       = torch.load(MODEL_PATH, map_location="cpu")
    input_dim  = ckpt["input_dim"]
    model      = build_model(input_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"✅ Loaded model from {MODEL_PATH}  (input_dim={input_dim})")
    return model, input_dim


# ── Status ────────────────────────────────────────────────────────────────────

def print_status():
    if not STATS_PATH.exists():
        print("No trained model found. Run stage7_classifier.py first.")
        return
    with open(STATS_PATH) as f:
        data = json.load(f)
    stats = data["stats"]
    print(f"\n📊 Classifier stats:")
    for k, v in stats.items():
        print(f"   {k:<12}: {v:.4f}")
    print(f"   input_dim  : {data['input_dim']}")
    history = data.get("history", [])
    if history:
        best = min(history, key=lambda x: x["val_loss"])
        print(f"   best epoch : {best['epoch']}  val_loss={best['val_loss']:.4f}  val_auc={best['val_auc']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    X, y = load_pairs()
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_data(X, y)

    print(f"\n   Train : {len(X_tr)}")
    print(f"   Val   : {len(X_val)}")
    print(f"   Test  : {len(X_te)}")

    input_dim = X.shape[1]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, history = train(X_tr, y_tr, X_val, y_val, input_dim)
    stats = evaluate(model, X_te, y_te, device)
    save_model(model, input_dim, stats, history)

    print("\n✅ Stage 7 complete.")
    print("   → Next: python stage8_graph.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",   action="store_true", help="Evaluate saved model on test set")
    parser.add_argument("--status", action="store_true", help="Show model stats and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.eval:
        import torch
        X, y     = load_pairs()
        _, _, _, _, X_te, y_te = split_data(X, y)
        model, _ = load_model()
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model    = model.to(device)
        evaluate(model, X_te, y_te, device)
    else:
        run()