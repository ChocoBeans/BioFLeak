"""Gradient inversion: recover a client's training feature vectors from its shared update.

Threat model: an honest-but-curious server that knows the architecture, the global model it sent
and the client's update. With one plain-SGD step, update = -lr * gradient. The attacker works in the
client's locally standardized feature space; mapping back to log2(TPM+1) would additionally need that
node's training mean/std (attacker knowledge, not used here).

- `closed_form` (batch size 1): for logistic regression grad_w = (p - y) x and grad_b = p - y, so
  x = grad_w / grad_b. For the MLP, first-layer row j has grad_W1[j] = d_j x and grad_b1[j] = d_j,
  so x = grad_W1[j] / grad_b1[j] for any active hidden unit j. Exact, no optimisation.
- `gradient_matching` (any batch size, DLG-style): optimise dummy inputs and labels until their
  gradient matches the observed one. For batch size > 1 the gradient is an average, so the solution
  is not unique.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from models import build_model


def param_slices(model):
    """[(name, shape, slice)] of each parameter inside the flat state vector."""
    out, start = [], 0
    for name, p in model.named_parameters():
        out.append((name, tuple(p.shape), slice(start, start + p.numel())))
        start += p.numel()
    return out


def closed_form(model_name, n_features, grad):
    """Exact reconstruction of the single sample behind a batch-size-1 gradient (flat vector)."""
    parts = {n: grad[s].reshape(shape) for n, shape, s in param_slices(build_model(model_name, n_features))}
    if model_name == "logreg":
        return parts["linear.weight"][0] / parts["linear.bias"][0]
    j = np.argmax(np.abs(parts["net.0.bias"]))  # the most active hidden unit (best conditioned)
    return parts["net.0.weight"][j] / parts["net.0.bias"][j]


def infer_label(grad):
    """Label of a batch-size-1 update: the output-bias gradient is p - y, negative iff y = 1."""
    return float(grad[-1] < 0)


def gradient_matching(model_name, n_features, global_vec, grad, batch_size, steps=500, lr=0.1, seed=0,
                      label=None):
    """DLG-style attack: returns a (batch_size, n_features) array of reconstructed samples.

    Soft labels are learned unless `label` (a known or inferred hard label, batch size 1) is given;
    without it x and -x with the opposite label produce the same gradient.
    """
    model = build_model(model_name, n_features)
    model.load_state_dict({n: torch.as_tensor(global_vec[s].reshape(shape), dtype=torch.float32)
                           for n, shape, s in param_slices(model)})
    params = list(model.parameters())
    target = torch.as_tensor(grad, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, n_features, generator=generator).requires_grad_()
    y_logit = torch.randn(batch_size, generator=generator).requires_grad_()  # soft label, learned
    opt = torch.optim.Adam([x, y_logit], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), torch.full((batch_size,), label) if label is not None
                       else torch.sigmoid(y_logit))
        g = torch.cat([t.flatten() for t in torch.autograd.grad(loss, params, create_graph=True)])
        ((g - target) ** 2).sum().backward()
        opt.step()
    return x.detach().numpy()


def cosine_matrix(a, b):
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return (a @ b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :] + 1e-12)


def matched_cosine(recon, truth):
    """Mean cosine similarity after the best one-to-one pairing of reconstructions to true samples."""
    c = cosine_matrix(recon, truth)
    rows, cols = linear_sum_assignment(-c)
    return float(c[rows, cols].mean())
