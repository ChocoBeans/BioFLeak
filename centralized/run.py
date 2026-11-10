"""Centralized baseline: pool the node training sets, train, evaluate on the shared test set.

`python -m centralized.run` trains logistic regression and a small MLP over several seeds for
MAX_EPOCHS epochs, logging validation and test metrics at every epoch. The epoch is then chosen
by early stopping on the pooled validation AUC (utils.train.select_best, the same rule as the
federated run), never from test numbers. Writes results/centralized/metrics.csv and the
per-epoch curves in results/centralized/epochs/. The data split is fixed (see utils.preprocess);
seeds only change model initialisation and batch order.

`--scaling local` scales every node with its own training statistics (as FedAvg does) while still
training on the pooled data; it isolates how much of the FedAvg gap comes from local scaling.
It writes to results/centralized_local/ so the baseline in results/centralized/ is untouched.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from models import build_model
from utils.dataset import load_processed
from utils.preprocess import ROOT
from utils.standardization import standardize_local, standardize_pooled
from utils.train import (BATCH_SIZE, LR, SEEDS, WEIGHT_DECAY, evaluate, loss_on, predict_proba,
                         select_best, train_epoch)

MAX_EPOCHS = 200
MODELS = ("logreg", "mlp")
METRICS = ["auc", "accuracy", "f1", "precision", "recall"]


def train_curve(name, train, val, test, seed):
    """Train for MAX_EPOCHS; return (one row per epoch, test probabilities per epoch).

    Columns follow the federated per-round files: train_loss, val_* (pooled validation set) and
    test_<split>_* for the pooled test set and each node. The test columns are for reporting and
    plotting only.
    """
    X_test, y_test, node_of = test
    splits = {"pooled": slice(None), **{n: node_of == n for n in sorted(set(node_of))}}
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = build_model(name, train[0].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rows, probs = [], []
    for epoch in range(1, MAX_EPOCHS + 1):
        row = {"epoch": epoch, "train_loss": train_epoch(model, *train, optimizer, BATCH_SIZE, generator)}
        v = evaluate(model, *val)
        row.update(val_auc=v["auc"], val_accuracy=v["accuracy"], val_loss=loss_on(model, *val))
        for split, mask in splits.items():
            for k, x in evaluate(model, X_test[mask], y_test[mask]).items():
                row[f"test_{split}_{k}"] = x
        rows.append(row)
        probs.append(predict_proba(model, X_test))
    return pd.DataFrame(rows), np.stack(probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scaling", choices=("pooled", "local"), default="pooled")
    scaling = parser.parse_args().scaling
    standardize = standardize_pooled if scaling == "pooled" else standardize_local
    test, nodes = standardize(*load_processed())
    pool = lambda split: tuple(np.concatenate(a) for a in zip(*[d[split] for d in nodes.values()]))
    train, val = pool("train"), pool("val")
    n_of = {"pooled": len(test[1]), **{n: int((test[2] == n).sum()) for n in sorted(nodes)}}

    out_dir = ROOT / "results" / ("centralized" if scaling == "pooled" else "centralized_local")
    (out_dir / "epochs").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    rows = []
    for name in MODELS:
        for seed in SEEDS:
            curve, probs = train_curve(name, train, val, test, seed)
            curve.to_csv(out_dir / "epochs" / f"{name}_s{seed}.csv", index=False)
            best = select_best(curve["val_auc"].tolist())
            chosen = curve[curve["epoch"] == best].iloc[0]
            np.save(out_dir / "predictions" / f"{name}_s{seed}.npy", probs[best - 1])  # test P(mutated), for paired tests
            for split, n in n_of.items():
                rows.append({"model": name, "seed": seed, "split": split, "n": n, "best_epoch": best,
                             **{m: chosen[f"test_{split}_{m}"] for m in METRICS}})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)

    print(f"Pooled test set ({len(SEEDS)} seeds, mean ± std)")
    for name, g in df[df.split == "pooled"].groupby("model"):
        print(f"  {name:7s}" + "  ".join(f"{m} {g[m].mean():.3f}±{g[m].std():.3f}" for m in METRICS)
              + f"  (best epoch {g.best_epoch.mean():.0f})")
    print("Per-node test AUC (mean ± std)")
    for split, g in df[df.split != "pooled"].groupby("split"):
        print(f"  {split:7s}(n={int(g.n.iloc[0]):3d}) " + "  ".join(
            f"{name} {h.auc.mean():.3f}±{h.auc.std():.3f}" for name, h in g.groupby("model")))
    print(f"saved {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
