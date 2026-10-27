"""Centralized baseline: pool the node training sets, train, evaluate on the shared test set.

`python -m centralized.run` trains logistic regression and a small MLP over several seeds, with
early stopping on validation AUC, and writes results/centralized/metrics.csv. The data split is
fixed (see utils.preprocess); seeds only change model initialisation and batch order.
"""
import copy

import numpy as np
import pandas as pd
import torch

from models import build_model
from utils.dataset import load_processed
from utils.preprocess import ROOT
from utils.standardization import standardize_pooled
from utils.train import BATCH_SIZE, LR, SEEDS, WEIGHT_DECAY, evaluate, train_epoch

MAX_EPOCHS, PATIENCE = 200, 20
MODELS = ("logreg", "mlp")
METRICS = ["auc", "accuracy", "f1", "precision", "recall"]


def fit(name, train, val, seed):
    """Train with early stopping on validation AUC; returns the best model and its epoch."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = build_model(name, train[0].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_auc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_epoch(model, *train, optimizer, BATCH_SIZE, generator)
        auc = evaluate(model, *val)["auc"]
        if auc > best_auc:
            best_auc, best_state, best_epoch = auc, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_epoch


def main():
    test, nodes = standardize_pooled(*load_processed())
    pool = lambda split: tuple(np.concatenate(a) for a in zip(*[d[split] for d in nodes.values()]))
    train, val = pool("train"), pool("val")

    rows = []
    for name in MODELS:
        for seed in SEEDS:
            model, epoch = fit(name, train, val, seed)
            for split, mask in {"pooled": slice(None), **{n: test[2] == n for n in nodes}}.items():
                X, y = test[0][mask], test[1][mask]
                rows.append({"model": name, "seed": seed, "split": split, "n": len(y),
                             "best_epoch": epoch, **evaluate(model, X, y)})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "centralized" / "metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Pooled test set ({len(SEEDS)} seeds, mean ± std)")
    for name, g in df[df.split == "pooled"].groupby("model"):
        print(f"  {name:7s}" + "  ".join(f"{m} {g[m].mean():.3f}±{g[m].std():.3f}" for m in METRICS)
              + f"  (best epoch {g.best_epoch.mean():.0f})")
    print("Per-node test AUC (mean ± std)")
    for split, g in df[df.split != "pooled"].groupby("split"):
        n = int(g.n.iloc[0])
        print(f"  {split:7s}(n={n:3d}) " + "  ".join(
            f"{name} {h.auc.mean():.3f}±{h.auc.std():.3f}" for name, h in g.groupby("model")))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
