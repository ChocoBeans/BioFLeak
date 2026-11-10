"""Privacy-utility cost of client-side DP: FedAvg with DP-SGD at several epsilons.

`python -m defenses.run_dp` -> results/defenses/dp_utility.csv and a summary table. Every client
spends the same (epsilon, delta) over a fixed round budget (so the model is the one after the last
round; early stopping on validation would spend extra privacy budget). epsilon = inf is non-private
SGD with the same learning rate, batch size and rounds, the like-for-like reference (the Adam baseline
of the federated run is a different optimizer).
"""
import argparse

import pandas as pd
from flwr.simulation import run_simulation

from federated.client_app import app as client_app
from federated.server_app import DEFAULTS, make_server_app, run_tag
from utils.dataset import load_processed
from utils.dp import CLIP
from utils.preprocess import ROOT
from utils.train import SEEDS

OUT = ROOT / "results" / "defenses"
METRICS = ["auc", "accuracy", "f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["logreg", "mlp"])
    parser.add_argument("--epsilons", type=float, nargs="*", default=[1, 3, 8, float("inf")])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=CLIP)
    parser.add_argument("--out", default=str(OUT / "dp_utility.csv"))
    args = parser.parse_args()

    test, nodes = load_processed()
    n_of = {"pooled": len(test[1]), **{n: int((test[2] == n).sum()) for n in sorted(nodes)}}
    rows = []
    for model in args.models:
        for eps in args.epsilons:
            for seed in args.seeds:
                cfg = {"model": model, "seed": seed, "optimizer": "sgd", "learning-rate": args.learning_rate,
                       "dp-epsilon": 0.0 if eps == float("inf") else eps, "dp-clip": args.clip,
                       "num-server-rounds": args.rounds, "out-dir": str(OUT / "rounds"),
                       "capture": True, "capture-dir": str(OUT / "captures")}  # reused by run_dp_membership
                run_simulation(server_app=make_server_app(cfg), client_app=client_app, num_supernodes=4)
                last = pd.read_csv(OUT / "rounds" / f"{run_tag({**DEFAULTS, **cfg})}.csv").iloc[-1]
                for split, n in n_of.items():
                    rows.append({"model": model, "epsilon": eps, "seed": seed, "split": split, "n": n,
                                 "val_auc": last["val_auc"], **{m: last[f"test_{split}_{m}"] for m in METRICS}})
                print(f"{model} eps={eps} seed={seed} test auc {last['test_pooled_auc']:.3f} "
                      f"val auc {last['val_auc']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df[df.split == "pooled"].groupby(["model", "epsilon"])[["auc", "accuracy", "val_auc"]]
          .agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
