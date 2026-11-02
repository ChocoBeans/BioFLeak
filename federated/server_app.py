"""Flower ServerApp: FedAvg over the CLL nodes.

Each round it logs (a) the client-side validation metrics, weighted-averaged by Flower, and
(b) the global model's score on the shared test set, pooled and per node (server-side, evaluation
only). Per-round results go to results/federated/rounds/<tag>.csv; federated/run.py picks the
round from the client-side validation AUC, never from the test numbers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # utils/ and models/ live at the repo root

import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from models import build_model
from utils.dataset import load_processed
from utils.standardization import standardize_local
from utils.train import BATCH_SIZE, LR, WEIGHT_DECAY, evaluate

DEFAULTS = {"model": "logreg", "seed": 0, "optimizer": "adam", "local-epochs": 1,
            "learning-rate": LR, "weight-decay": WEIGHT_DECAY, "batch-size": BATCH_SIZE,
            "num-server-rounds": 200, "out-dir": str(ROOT / "results" / "federated" / "rounds")}


def run_tag(cfg):
    return (f"{cfg['model']}_{cfg['optimizer']}_e{cfg['local-epochs']}_b{cfg['batch-size']}"
            f"_s{cfg['seed']}")


def run_fedavg(grid: Grid, cfg: dict) -> None:
    # Simulation shortcut: the held-out test set is stored server-side for evaluation only, and
    # each test patient is scaled with their own cohort's training statistics.
    test, nodes = standardize_local(*load_processed())
    X_test, y_test, node_of = test
    names = sorted(nodes)

    def test_metrics(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = build_model(cfg["model"], X_test.shape[1])
        model.load_state_dict(arrays.to_torch_state_dict())
        out = {}
        for split, mask in {"pooled": slice(None), **{n: node_of == n for n in names}}.items():
            for k, v in evaluate(model, X_test[mask], y_test[mask]).items():
                out[f"test_{split}_{k}"] = float(v)
        return MetricRecord(out)

    torch.manual_seed(cfg["seed"])
    initial = ArrayRecord(build_model(cfg["model"], X_test.shape[1]).state_dict())
    strategy = FedAvg(min_train_nodes=len(names), min_evaluate_nodes=len(names),
                      min_available_nodes=len(names))
    train_config = ConfigRecord({
        "model": cfg["model"], "optimizer": cfg["optimizer"], "local-epochs": cfg["local-epochs"],
        "lr": cfg["learning-rate"], "weight-decay": cfg["weight-decay"],
        "batch-size": cfg["batch-size"], "seed": cfg["seed"]})
    result = strategy.start(grid=grid, initial_arrays=initial, num_rounds=cfg["num-server-rounds"],
                            train_config=train_config,
                            evaluate_config=ConfigRecord({"model": cfg["model"]}),
                            evaluate_fn=test_metrics)

    rows = []
    for r in range(1, cfg["num-server-rounds"] + 1):
        row = {"round": r}
        for source in (result.train_metrics_clientapp, result.evaluate_metrics_clientapp,
                       result.evaluate_metrics_serverapp):
            row.update(dict(source.get(r, {})))
        rows.append(row)
    out = Path(cfg["out-dir"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{run_tag(cfg)}.csv", index=False)


def make_server_app(cfg: dict | None = None) -> ServerApp:
    """A ServerApp that runs FedAvg with `cfg` (missing keys fall back to DEFAULTS)."""
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        run_fedavg(grid, {**DEFAULTS, **(cfg or dict(context.run_config))})

    return app
