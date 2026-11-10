"""Choose the SGD learning rate for the DP / secure-aggregation runs on validation AUC.

`python -m defenses.tune_lr` -> results/defenses/lr_tuning.csv. Logistic regression, FedAvg with
SGD, batch 32, 50 rounds, at epsilon 8 and no DP, 2 seeds, for each candidate learning rate. The
choice uses only the clients' validation AUC after the last round (never test numbers). The per-round
logs go to results/defenses/lr_tuning/ because run tags do not contain the learning rate and would
otherwise overwrite the real runs. The grid picked 0.1 (the default in defenses/run_dp.py): it ties 0.02 at epsilon 8 and is within
0.002 of the best without DP, so it is a compromise. This tuning is not charged to epsilon.
"""
import argparse

import pandas as pd
from flwr.simulation import run_simulation

from federated.client_app import app as client_app
from federated.server_app import DEFAULTS, make_server_app, run_tag
from utils.preprocess import ROOT

OUT = ROOT / "results" / "defenses"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rates", type=float, nargs="*", default=[0.02, 0.1, 0.5])
    parser.add_argument("--epsilons", type=float, nargs="*", default=[8, float("inf")])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()
    rows = []
    for lr in args.learning_rates:
        for eps in args.epsilons:
            for seed in args.seeds:
                cfg = {"model": "logreg", "seed": seed, "optimizer": "sgd", "learning-rate": lr,
                       "dp-epsilon": 0.0 if eps == float("inf") else eps, "num-server-rounds": args.rounds,
                       "out-dir": str(OUT / "lr_tuning" / f"lr{lr}")}
                run_simulation(server_app=make_server_app(cfg), client_app=client_app, num_supernodes=4)
                last = pd.read_csv(OUT / "lr_tuning" / f"lr{lr}" / f"{run_tag({**DEFAULTS, **cfg})}.csv").iloc[-1]
                rows.append({"learning_rate": lr, "epsilon": eps, "seed": seed, "val_auc": last["val_auc"]})
                print(f"lr={lr} eps={eps} seed={seed} val auc {last['val_auc']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lr_tuning.csv", index=False)
    print(df.groupby(["learning_rate", "epsilon"])["val_auc"].mean().round(4).unstack())


if __name__ == "__main__":
    main()
