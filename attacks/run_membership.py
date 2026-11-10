"""Score the membership-inference attack on the captured baseline runs
(`python -m attacks.run_capture --membership` first).

Members of node k = its training patients; non-members = the patients of node k's validation and test
sets (never trained on). Each candidate is scored in node k's locally standardized space. Attacked
models: the round chosen by early stopping (as in the federated baseline) and the last round (200,
trained longer so more overfit). The score of each patient is averaged over the 5 seeds before the AUC
is taken, with a bootstrap 95% interval over patients; per-seed AUCs give the seed spread.

`python -m attacks.run_membership` -> results/attacks/membership.csv and a summary table.
"""
import numpy as np
import pandas as pd

from attacks.membership import (attack_auc, bootstrap_ci, model_from_vector, per_patient,
                                sample_loss)
from federated.server_app import DEFAULTS, run_tag
from utils.dataset import load_patients, load_processed
from utils.preprocess import ROOT
from utils.standardization import standardize_local
from utils.train import SEEDS, select_best

OUT = ROOT / "results" / "attacks"
MODELS = ("logreg", "mlp")


def main():
    test, nodes = standardize_local(*load_processed())
    X_test, y_test, node_of = test
    test_ids, ids = load_patients()
    names = sorted(nodes)
    n_features = X_test.shape[1]
    n_train = np.array([len(nodes[n]["train"][1]) for n in names], dtype=float)

    # candidates per node: (X, y, patient ids, is_member)
    cand = {}
    for n in names:
        m = node_of == n
        X = np.vstack([nodes[n]["train"][0], nodes[n]["val"][0], X_test[m]])
        y = np.concatenate([nodes[n]["train"][1], nodes[n]["val"][1], y_test[m]])
        pid = np.concatenate([ids[n]["train"], ids[n]["val"], test_ids[m]])
        member = np.r_[np.ones(len(ids[n]["train"]), bool), np.zeros(len(ids[n]["val"]) + m.sum(), bool)]
        cand[n] = (X, y, pid, member)

    scores = {}  # (model, which round, attack, node) -> list over seeds of (patient ids, score, member)
    for model_name in MODELS:
        for seed in SEEDS:
            cfg = {**DEFAULTS, "model": model_name, "seed": seed}
            cap = np.load(OUT / "captures" / f"{run_tag(cfg)}.npz")
            rounds_df = pd.read_csv(OUT / "rounds" / f"{run_tag(cfg)}.csv")
            best = select_best(rounds_df["val_auc"].tolist())
            last = int(cap["rounds"][-1])
            for which, r in (("selected", best), ("last", last)):
                i = int(np.where(cap["rounds"] == r)[0][0])
                sent = cap["glob"][i]  # global model the clients received in round r
                after = sent + (n_train[:, None] * cap["delta"][i]).sum(0) / n_train.sum()  # FedAvg
                g_after, g_sent = (model_from_vector(model_name, n_features, v) for v in (after, sent))
                for k, n in enumerate(names):
                    X, y, pid, member = cand[n]
                    local = model_from_vector(model_name, n_features, sent + cap["delta"][i, k])
                    for attack, s in (("global_loss", -sample_loss(g_after, X, y)),
                                      ("update_gain", sample_loss(g_sent, X, y) - sample_loss(local, X, y))):
                        p, score = per_patient(s, pid)
                        mem = np.isin(p, np.unique(pid[member]))
                        scores.setdefault((model_name, which, attack, n), []).append((p, score, mem, r))

    rows = []
    for (model_name, which, attack, n), runs in scores.items():
        patients, _, member, _ = runs[0]
        per_seed = [attack_auc(sc[mem], sc[~mem]) for _, sc, mem, _ in runs]
        mean_score = np.mean([sc for _, sc, _, _ in runs], axis=0)  # same patient order in every seed
        lo, hi = bootstrap_ci(mean_score[member], mean_score[~member])
        rows.append({"model": model_name, "round": which, "attack": attack, "node": n,
                     "n_members": int(member.sum()), "n_nonmembers": int((~member).sum()),
                     "auc_mean_over_seeds": np.mean(per_seed), "auc_std_over_seeds": np.std(per_seed),
                     "auc_of_seed_mean_score": attack_auc(mean_score[member], mean_score[~member]),
                     "ci_low": lo, "ci_high": hi, "rounds": [r for *_, r in runs]})
    # all nodes together: concatenate the per-node, seed-averaged scores
    for model_name in MODELS:
        for which in ("selected", "last"):
            for attack in ("global_loss", "update_gain"):
                parts = [(np.mean([sc for _, sc, _, _ in scores[(model_name, which, attack, n)]], axis=0),
                          scores[(model_name, which, attack, n)][0][2]) for n in names]
                s, m = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
                lo, hi = bootstrap_ci(s[m], s[~m])
                rows.append({"model": model_name, "round": which, "attack": attack, "node": "all",
                             "n_members": int(m.sum()), "n_nonmembers": int((~m).sum()),
                             "auc_of_seed_mean_score": attack_auc(s[m], s[~m]), "ci_low": lo, "ci_high": hi})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "membership.csv", index=False)
    show = df.drop(columns=["rounds"], errors="ignore").round(3)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
