"""Training and evaluation shared by the centralized and federated runs."""
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

# Shared by the centralized and federated runs so the comparison is like for like.
LR, WEIGHT_DECAY, BATCH_SIZE = 1e-3, 1e-4, 32
SEEDS = range(5)  # vary model initialisation and batch order; the data split is fixed
PATIENCE = 20  # early stopping: stop after this many steps (epochs / rounds) without improvement


def select_best(val_auc, patience=PATIENCE):
    """Step (1-based epoch or round) chosen by early stopping on a validation-AUC series.

    Same result as stopping online: take the best step, and stop once `patience` further steps
    bring no improvement.
    """
    best, best_step = -1.0, 0
    for step, auc in enumerate(val_auc, start=1):
        if auc > best:
            best, best_step = auc, step
        elif step - best_step >= patience:
            break
    return best_step


def batch_generator(seed, server_round, partition_id):
    """Per-client, per-round RNG for batch order; the attacks recompute it to know the true batches."""
    return torch.Generator().manual_seed(seed * 1_000_003 + server_round * 101 + partition_id)


def sample_batches(n, steps, batch_size, generator=None):
    """Index tensors of `steps` mini-batches, each drawn without replacement from range(n)."""
    return [torch.randperm(n, generator=generator)[:batch_size] for _ in range(steps)]


def train_steps(model, X, y, optimizer, steps, batch_size, generator=None):
    """`steps` optimizer steps on random mini-batches (one step with plain SGD = -lr * gradient)."""
    model.train()
    X, y = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    for idx in sample_batches(len(X), steps, batch_size, generator):
        optimizer.zero_grad()
        loss = loss_fn(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / steps


def train_epoch(model, X, y, optimizer, batch_size=BATCH_SIZE, generator=None):
    """One shuffled pass over (X, y); returns the mean training loss."""
    model.train()
    X, y = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    for idx in torch.randperm(len(X), generator=generator).split(batch_size):
        optimizer.zero_grad()
        loss = loss_fn(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    return total / len(X)


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(X))).numpy()


@torch.no_grad()
def loss_on(model, X, y):
    """Mean binary cross-entropy of the model on (X, y)."""
    model.eval()
    return nn.BCEWithLogitsLoss()(model(torch.as_tensor(X)), torch.as_tensor(y, dtype=torch.float32)).item()


def evaluate(model, X, y, threshold=0.5):
    """AUC-ROC plus accuracy, F1, precision and recall (IGHV-mutated = positive class)."""
    p = predict_proba(model, X)
    pred = (p >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }
