"""Membership inference: was this patient in a node's training set?

Loss-threshold attack (Yeom et al.): training records tend to have lower loss than unseen ones, so
the score is -loss of the candidate (features and true label) under a model. The attacker is a server
that knows the candidate record and label and sees the global model and each client's returned model.
Two scores, both "higher = more likely a member of node k":
- `global_loss`: -loss under the global model after the round. Cohorts are disjoint, so a member of
  the federation is a member of exactly one node.
- `update_gain`: loss under the global model sent minus loss under node k's returned local model, i.e.
  how much node k's local training improved on the candidate. Specific to node k.
Metric: AUC of the score for members vs non-members (0.5 = guessing), computed per patient (a
patient with two samples in a node's train set is one member; their losses are averaged).
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from attacks.inversion import param_slices
from models import build_model


def model_from_vector(model_name, n_features, vec):
    model = build_model(model_name, n_features)
    model.load_state_dict({n: torch.as_tensor(vec[s].reshape(shape), dtype=torch.float32)
                           for n, shape, s in param_slices(model)})
    return model.eval()


@torch.no_grad()
def sample_loss(model, X, y):
    """Per-sample binary cross-entropy."""
    z = model(torch.as_tensor(X)).numpy().astype(np.float64)
    return np.logaddexp(0.0, z) - np.asarray(y) * z


def per_patient(values, patients):
    """Mean of `values` per patient; returns (patient ids, means)."""
    ids, inverse = np.unique(patients, return_inverse=True)
    return ids, np.bincount(inverse, weights=values) / np.bincount(inverse)


def attack_auc(member_scores, nonmember_scores):
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    return roc_auc_score(labels, scores)


def bootstrap_ci(member_scores, nonmember_scores, n=1000, seed=0):
    """95% interval of the AUC when resampling patients."""
    rng = np.random.default_rng(seed)
    aucs = [attack_auc(rng.choice(member_scores, len(member_scores)),
                       rng.choice(nonmember_scores, len(nonmember_scores))) for _ in range(n)]
    return tuple(np.percentile(aucs, [2.5, 97.5]))
