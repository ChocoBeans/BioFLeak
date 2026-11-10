"""Run FedAvg and record every client update (the honest-but-curious server's view).

`python -m attacks.run_capture` -> results/attacks/captures/<tag>.npz (one per model x batch size x seed).
With plain SGD and one mini-batch step per round, each client update is exactly -lr * (gradient of the
batch at the global model), which is what the gradient-inversion attack reads. The learning rate is
larger than the baseline's (0.1 instead of 1e-3) so the model actually trains in the few steps a round
allows.

`python -m attacks.run_capture --membership` instead records the standard baseline protocol (Adam,
1 local epoch, batch 32, 200 rounds, 5 seeds), the setting whose trained models the membership
inference attack targets.
"""
import argparse

from flwr.simulation import run_simulation

from federated.client_app import app as client_app
from federated.server_app import make_server_app
from utils.preprocess import ROOT

OUT = ROOT / "results" / "attacks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["logreg", "mlp"])
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--membership", action="store_true",
                        help="capture the baseline Adam protocol instead (for membership inference)")
    args = parser.parse_args()
    if args.membership:
        for model in args.models:
            for seed in range(5):
                cfg = {"model": model, "seed": seed, "capture": True,  # other keys: baseline defaults
                       "capture-dir": str(OUT / "captures"), "out-dir": str(OUT / "rounds")}
                run_simulation(server_app=make_server_app(cfg), client_app=client_app, num_supernodes=4)
        return
    for model in args.models:
        for batch in args.batch_sizes:
            for seed in args.seeds:
                cfg = {"model": model, "seed": seed, "optimizer": "sgd", "local-steps": 1,
                       "batch-size": batch, "learning-rate": args.learning_rate,
                       "num-server-rounds": args.rounds, "capture": True,
                       "capture-dir": str(OUT / "captures"), "out-dir": str(OUT / "rounds")}
                run_simulation(server_app=make_server_app(cfg), client_app=client_app, num_supernodes=4)


if __name__ == "__main__":
    main()
