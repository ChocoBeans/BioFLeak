"""Paired bootstrap for differences in test AUC between two training setups.

All setups are scored on the same 140 test samples, so a difference in AUC is a paired quantity:
`python -m utils.uncertainty` resamples test patients (with replacement, a patient's samples move
together), recomputes each setup's AUC on every resample, averages over the seeds, and reports the
mean difference with a 95% interval. This is much tighter than comparing two separate standard
errors (about 0.017 each at n = 140), because both setups see the same hard patients.

Test probabilities come from:
- centralized: results/centralized[_local]/predictions/ (written by `python -m centralized.run`);
- FedAvg: rebuilt from the saved captures (global model + the weighted client updates of the round
  chosen by `select_best` on the logged validation AUC, or the last round for the fixed-round SGD runs).
  `rebuild_check` confirms the rebuilt model reproduces the logged test AUC.
Writes results/uncertainty.csv and prints the table.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from attacks.membership import model_from_vector
from utils.dataset import load_patients, load_processed
from utils.preprocess import ROOT
from federated.server_app import check_run_config
from utils.standardization import standardize_local
from utils.train import SEEDS, predict_proba, select_best

RESULTS = ROOT / "results"
MODELS = ("logreg", "mlp")
N_BOOT = 2000


def fedavg_probs(capture, model_name, X_test, weights, round_):
    """Test P(mutated) of the FedAvg global model after `round_` (1-based), from a capture file."""
    d = np.load(capture)
    delta = np.nan_to_num(d["delta"][round_ - 1])  # (clients, params); NaN = client did not reply
    w = np.asarray(weights, dtype=float)
    vec = d["glob"][round_ - 1] + (w[:, None] * delta).sum(0) / w.sum()
    return predict_proba(model_from_vector(model_name, X_test.shape[1], vec), X_test)


def paired_bootstrap(y, probs_a, probs_b, patients, n_boot=N_BOOT, seed=0):
    """Mean AUC(a) - AUC(b) over seeds, with a 95% interval over resampled test patients.

    probs_a / probs_b: arrays (seeds, n_test). Returns (mean_a, mean_b, diff, lo, hi).
    """
    ids, inverse = np.unique(patients, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(ids))]
    rng = np.random.default_rng(seed)
    point = lambda p: np.mean([roc_auc_score(y, s) for s in p])
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))])
        if len(set(y[idx])) < 2:
            continue
        diffs.append(np.mean([roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
                              for a, b in zip(probs_a, probs_b)]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point(probs_a), point(probs_b), point(probs_a) - point(probs_b), lo, hi


def main():
    test_ids, nodes_ids = load_patients()
    raw_test, raw_nodes = load_processed()
    (X_test, y_test, _), _ = standardize_local(raw_test, raw_nodes)
    sizes = [len(raw_nodes[n]["train"][1]) for n in sorted(raw_nodes)]

    # --- Part 1: Adam protocol, round/epoch chosen on validation AUC -------------------------------
    central, central_local, fed = {}, {}, {}
    for m in MODELS:
        central[m] = np.stack([np.load(RESULTS / "centralized" / "predictions" / f"{m}_s{s}.npy") for s in SEEDS])
        central_local[m] = np.stack([np.load(RESULTS / "centralized_local" / "predictions" / f"{m}_s{s}.npy")
                                     for s in SEEDS])
        rows = []
        for s in SEEDS:
            tag = f"{m}_adam_e1_b32_s{s}"
            log = pd.read_csv(RESULTS / "federated" / "rounds" / f"{tag}.csv")
            r = select_best(log["val_auc"].tolist())
            p = fedavg_probs(RESULTS / "attacks" / "captures" / f"{tag}.npz", m, X_test, sizes, r)
            logged = log.loc[log["round"] == r, "test_pooled_auc"].iloc[0]
            assert abs(roc_auc_score(y_test, p) - logged) < 1e-4, (tag, roc_auc_score(y_test, p), logged)
            rows.append(p)
        fed[m] = np.stack(rows)

    out = []
    for m in MODELS:
        for label, a, b in [("FedAvg vs centralized (pooled scaling)", fed[m], central[m]),
                            ("FedAvg vs centralized (local scaling)", fed[m], central_local[m]),
                            ("centralized local vs pooled scaling", central_local[m], central[m])]:
            out.append({"part": 1, "model": m, "comparison": label, **dict(zip(
                ["auc_a", "auc_b", "diff", "ci_low", "ci_high"], paired_bootstrap(y_test, a, b, test_ids)))})

    # --- Part 2: SGD protocol, DP vs no DP at the last of 50 rounds --------------------------------
    sgd = {}
    for m in MODELS:
        for eps in (1, 3, 8, "inf"):
            rows = []
            for s in SEEDS:
                tag = f"{m}_sgd_e1_b32_s{s}" if eps == "inf" else f"{m}_dp{eps}_sgd_e1_b32_s{s}"
                csv = RESULTS / "defenses" / "rounds" / f"{tag}.csv"
                check_run_config(csv, **{"learning-rate": 0.1, "num-server-rounds": 50})
                log = pd.read_csv(csv)
                p = fedavg_probs(RESULTS / "defenses" / "captures" / f"{tag}.npz", m, X_test, sizes, int(log["round"].iloc[-1]))
                assert abs(roc_auc_score(y_test, p) - log["test_pooled_auc"].iloc[-1]) < 1e-4, tag
                rows.append(p)
            sgd[m, eps] = np.stack(rows)
    for m in MODELS:
        for eps in (1, 3, 8):
            out.append({"part": 2, "model": m, "comparison": f"DP epsilon {eps} vs no DP (SGD)", **dict(zip(
                ["auc_a", "auc_b", "diff", "ci_low", "ci_high"],
                paired_bootstrap(y_test, sgd[m, eps], sgd[m, "inf"], test_ids)))})

    df = pd.DataFrame(out)
    df.to_csv(RESULTS / "uncertainty.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"saved {RESULTS / 'uncertainty.csv'}  ({N_BOOT} resamples of test patients; a = first setup)")


if __name__ == "__main__":
    main()
