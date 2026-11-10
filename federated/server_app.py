"""Flower ServerApp: FedAvg over the CLL nodes.

Each round it logs (a) the client-side validation metrics, weighted-averaged by Flower, and
(b) the global model's score on the shared test set, pooled and per node (server-side, evaluation
only). Per-round results go to results/federated/rounds/<tag>.csv; federated/run.py picks the
round from the client-side validation AUC, never from the test numbers.
"""
import sys
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # utils/ and models/ live at the repo root

import numpy as np
import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from federated.capture_strategy import CaptureFedAvg
from models import build_model
from utils.dataset import load_processed
from utils.standardization import standardize_local
from utils.dp import CLIP, DELTA
from utils.train import BATCH_SIZE, LR, WEIGHT_DECAY, evaluate

DEFAULTS = {"model": "logreg", "seed": 0, "optimizer": "adam", "local-epochs": 1,
            "learning-rate": LR, "weight-decay": WEIGHT_DECAY, "batch-size": BATCH_SIZE,
            "local-steps": 0, "capture": False, "dp-epsilon": 0.0, "dp-delta": DELTA, "dp-clip": CLIP,
            "num-server-rounds": 200, "capture-dir": str(ROOT / "results" / "attacks" / "captures"),
            "out-dir": str(ROOT / "results" / "federated" / "rounds")}


def run_tag(cfg):
    steps = f"_k{cfg['local-steps']}" if cfg["local-steps"] else ""  # keeps the baseline tags unchanged
    dp = f"_dp{cfg['dp-epsilon']:g}" if cfg["dp-epsilon"] > 0 else ""
    return (f"{cfg['model']}{dp}_{cfg['optimizer']}_e{cfg['local-epochs']}{steps}_b{cfg['batch-size']}"
            f"_s{cfg['seed']}")


SIDECAR_KEYS = ("learning-rate", "weight-decay", "num-server-rounds", "dp-clip", "dp-delta")  # not in run_tag


def check_run_config(path, **expected):
    """Fail if a run's recorded settings disagree with `expected` (e.g. learning-rate, num-server-rounds).

    Run tags omit the learning rate and the round budget, so a rerun with other settings overwrites
    the same file name. Settings come from the sidecar `<tag>.json` next to the file; for a capture
    (.npz) from before sidecars existed they are read from the file itself (it stores lr and rounds).
    Anything else without a sidecar is skipped.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        saved = json.loads(sidecar.read_text())
    elif path.suffix == ".npz":
        d = np.load(path)
        saved = {"learning-rate": float(d["lr"]), "num-server-rounds": len(d["rounds"])}
    else:
        return
    same = lambda a, b: math.isclose(a, b, rel_tol=1e-9) if isinstance(a, float) or isinstance(b, float) else a == b
    bad = {k: (saved.get(k), v) for k, v in expected.items() if k in saved and not same(saved[k], v)}
    if bad:
        raise ValueError(f"{path.name}: settings differ from what the reader expects (saved, expected): {bad}")


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
    strategy = (CaptureFedAvg if cfg["capture"] else FedAvg)(
        min_train_nodes=len(names), min_evaluate_nodes=len(names), min_available_nodes=len(names))
    train_config = ConfigRecord({
        "model": cfg["model"], "optimizer": cfg["optimizer"], "local-epochs": cfg["local-epochs"],
        "local-steps": cfg["local-steps"], "capture": cfg["capture"],
        "dp-epsilon": cfg["dp-epsilon"], "dp-delta": cfg["dp-delta"], "dp-clip": cfg["dp-clip"],
        "num-server-rounds": cfg["num-server-rounds"],
        "lr": cfg["learning-rate"], "weight-decay": cfg["weight-decay"],
        "batch-size": cfg["batch-size"], "seed": cfg["seed"]})
    result = strategy.start(grid=grid, initial_arrays=initial, num_rounds=cfg["num-server-rounds"],
                            train_config=train_config,
                            evaluate_config=ConfigRecord({"model": cfg["model"]}),
                            evaluate_fn=test_metrics)

    if cfg["capture"]:
        strategy.save(Path(cfg["capture-dir"]) / f"{run_tag(cfg)}.npz", lr=cfg["learning-rate"])
        Path(cfg["capture-dir"]).joinpath(f"{run_tag(cfg)}.json").write_text(
            json.dumps({k: cfg[k] for k in SIDECAR_KEYS}))
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
    (out / f"{run_tag(cfg)}.json").write_text(json.dumps({k: cfg[k] for k in SIDECAR_KEYS}))


def make_server_app(cfg: dict | None = None) -> ServerApp:
    """A ServerApp that runs FedAvg with `cfg` (missing keys fall back to DEFAULTS)."""
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        run_fedavg(grid, {**DEFAULTS, **(cfg or dict(context.run_config))})

    return app
