"""Flower ClientApp: one CLL node (cohort) trains and validates on its own data only.

Each client standardizes with the statistics of its own training data (utils.standardization).
Hyperparameters arrive in the per-round config message sent by the server.
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # utils/ and models/ live at the repo root

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from models import build_model
from utils.dataset import load_processed
from utils.standardization import standardize_local
from utils.train import evaluate, loss_on, train_epoch

app = ClientApp()


@lru_cache(maxsize=1)
def _nodes():
    """{node: {"train": (X, y), "val": (X, y)}}, locally standardized."""
    return standardize_local(*load_processed())[1]


def _local_data(context):
    names = sorted(_nodes())
    return _nodes()[names[int(context.node_config["partition-id"])]]


def _global_model(msg, n_features):
    model = build_model(msg.content["config"]["model"], n_features)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model


@app.train()
def train(msg: Message, context: Context):
    """Train the received global model on this node's training data."""
    cfg = msg.content["config"]
    X, y = _local_data(context)["train"]
    model = _global_model(msg, X.shape[1])
    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight-decay"])
    else:  # plain SGD: one local step then equals -lr * gradient (used for the attack runs)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    generator = torch.Generator().manual_seed(
        cfg["seed"] * 1_000_003 + cfg["server-round"] * 101 + int(context.node_config["partition-id"]))
    losses = [train_epoch(model, X, y, optimizer, cfg["batch-size"], generator)
              for _ in range(cfg["local-epochs"])]
    metrics = {"train_loss": sum(losses) / len(losses), "num-examples": len(y)}
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_global(msg: Message, context: Context):
    """Validate the received global model on this node's validation data (data never leaves)."""
    X, y = _local_data(context)["val"]
    model = _global_model(msg, X.shape[1])
    m = evaluate(model, X, y)
    metrics = {"val_auc": m["auc"], "val_accuracy": m["accuracy"], "val_loss": loss_on(model, X, y),
               "num-examples": len(y)}
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)
