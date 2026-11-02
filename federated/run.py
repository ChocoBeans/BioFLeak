"""Federated baseline: FedAvg across the four nodes, evaluated like the centralized run.

`python -m federated.run` simulates FedAvg for logistic regression and the small MLP over several
seeds (Flower simulation, in-process), picks the round for each run from the client-side
validation AUC (early stopping with the same patience as centralized, never using test numbers),
and writes results/federated/metrics.csv in the same columns as results/centralized/metrics.csv.

Settings can be changed with --optimizer (adam | sgd), --local-epochs and --batch-size; the attack
runs use --optimizer sgd --local-epochs 1 --batch-size 1..8.
"""
import argparse

import pandas as pd
from flwr.simulation import run_simulation

from federated.client_app import app as client_app
from federated.server_app import DEFAULTS, make_server_app, run_tag
from utils.dataset import load_processed
from utils.preprocess import ROOT
from utils.train import SEEDS

MODELS = ("logreg", "mlp")
MAX_ROUNDS, PATIENCE = 200, 20  # same budget and patience as the centralized run
METRICS = ["auc", "accuracy", "f1", "precision", "recall"]


def select_round(val_auc, patience=PATIENCE):
    """Round (1-based) chosen by early stopping on the client-side validation AUC series."""
    best, best_round = -1.0, 0
    for r, auc in enumerate(val_auc, start=1):
        if auc > best:
            best, best_round = auc, r
        elif r - best_round >= patience:
            break
    return best_round


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default=DEFAULTS["optimizer"], choices=["adam", "sgd"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULTS["local-epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch-size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning-rate"])
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--out", default=str(ROOT / "results" / "federated" / "metrics.csv"))
    args = parser.parse_args()

    test, nodes = load_processed()
    n_of = {"pooled": len(test[1]), **{n: int((test[2] == n).sum()) for n in sorted(nodes)}}
    rows = []
    for model in args.models:
        for seed in args.seeds:
            cfg = {"model": model, "seed": seed, "optimizer": args.optimizer,
                   "local-epochs": args.local_epochs, "batch-size": args.batch_size,
                   "learning-rate": args.learning_rate, "num-server-rounds": args.rounds}
            run_simulation(server_app=make_server_app(cfg), client_app=client_app,
                           num_supernodes=len(nodes))
            rounds = pd.read_csv(f"{DEFAULTS['out-dir']}/{run_tag({**DEFAULTS, **cfg})}.csv")
            best = select_round(rounds["val_auc"].tolist())
            row = rounds[rounds["round"] == best].iloc[0]
            for split, n in n_of.items():
                rows.append({"model": model, "seed": seed, "split": split, "n": n, "best_round": best,
                             **{m: row[f"test_{split}_{m}"] for m in METRICS},
                             "optimizer": args.optimizer, "local_epochs": args.local_epochs,
                             "batch_size": args.batch_size})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
