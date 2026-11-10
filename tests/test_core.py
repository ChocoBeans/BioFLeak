"""Fast checks of the claims the results rest on. Run from the repo root: `python -m pytest tests`.

Needs data/processed/ (`python -m utils.preprocess`) for the split test only.
"""
import numpy as np
import pytest
import torch

from attacks.inversion import closed_form
from defenses.run_secagg import mask_updates
from models import build_model
from utils.dataset import load_patients
from utils.preprocess import PROCESSED_DIR


def one_step_update(model_name, x, y, n_features=20, lr=0.1, seed=0):
    """Update of one plain-SGD step on a single sample: -lr * gradient, flattened in parameter order."""
    torch.manual_seed(seed)
    model = build_model(model_name, n_features)
    loss = torch.nn.BCEWithLogitsLoss()(model(torch.as_tensor(x)[None]), torch.tensor([float(y)]))
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return -lr * torch.cat([g.flatten() for g in grads]).detach().double().numpy()


@pytest.mark.parametrize("model_name", ["logreg", "mlp"])
@pytest.mark.parametrize("label", [0, 1])
def test_closed_form_recovers_single_sample(model_name, label):
    x = np.random.default_rng(1).normal(size=20).astype(np.float32)
    recon = closed_form(model_name, 20, one_step_update(model_name, x, label))
    assert np.allclose(recon, x, atol=1e-3)


def test_masks_cancel_in_the_sum_and_hide_each_update():
    rng = np.random.default_rng(0)
    deltas = rng.normal(0, 0.01, (4, 50))
    masked = mask_updates(deltas, rng)
    assert np.allclose(masked.sum(0), deltas.sum(0), atol=1e-10)
    assert not np.allclose(masked[0], deltas[0], atol=0.1)


def test_no_patient_in_two_splits():
    if not PROCESSED_DIR.exists():
        pytest.skip("run python -m utils.preprocess first")
    test_ids, nodes = load_patients()
    sets = [set(test_ids)] + [set(ids) for d in nodes.values() for ids in d.values()]
    sizes = [len(s) for s in sets]
    assert len(set().union(*sets)) == sum(sizes), "a patient appears in more than one split or node"
