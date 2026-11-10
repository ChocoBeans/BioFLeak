"""Score the membership-inference attack on captured runs.

`python -m attacks.run_membership` scores the baseline protocol (`python -m attacks.run_capture
--membership` first); `defenses/run_dp_membership.py` reuses `membership_table` for the DP runs.

Members of node k = its training patients; non-members = the patients of node k's validation and test
sets (never trained on). Each candidate is scored in node k's locally standardized space. Attacked
models: the round chosen by early stopping (as in the federated baseline) and the last round (200,
trained longer so more overfit). The score of each patient is averaged over the seeds before the AUC
is taken, with a bootstrap 95% interval over patients; per-seed AUCs give the seed spread.

Output: results/attacks/membership.csv and a summary table.
"""
from pathlib import Path

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
ATTACKS = ("global_loss", "update_gain")


def membership_table(settings, models=MODELS, seeds=SEEDS):
    """Attack AUCs for every setting; a setting is {name, cfg, capture_dir, rounds_dir, which}.

    `cfg` overrides the federated DEFAULTS to find the captured runs; `which` lists the rounds
    attacked ("selected" = early-stopped on validation AUC, "last" = final round).
    """
    test, nodes = standardize_local(*load_processed())
    X_test, y_test, node_of = test
    test_ids, ids = load_patients()
    names = sorted(nodes)
    n_features = X_test.shape[1]
    n_train = np.array([len(nodes[n]["train"][1]) for n in names], dtype=float)

    cand = {}  # node -> (X, y, patient ids, is_member) of its train, val and test patients
    for n in names:
        m = node_of == n
        X = np.vstack([nodes[n]["train"][0], nodes[n]["val"][0], X_test[m]])
        y = np.concatenate([nodes[n]["train"][1], nodes[n]["val"][1], y_test[m]])
        pid = np.concatenate([ids[n]["train"], ids[n]["val"], test_ids[m]])
        member = np.r_[np.ones(len(ids[n]["train"]), bool), np.zeros(len(ids[n]["val"]) + m.sum(), bool)]
        cand[n] = (X, y, pid, member)

    scores = {}  # (setting, model, round kind, attack, node) -> per seed (patients, score, is_member, round)
    for st in settings:
        for model_name in models:
            for seed in seeds:
                cfg = {**DEFAULTS, **st["cfg"], "model": model_name, "seed": seed}
                cap = np.load(Path(st["capture_dir"]) / f"{run_tag(cfg)}.npz")
                val_auc = pd.read_csv(Path(st["rounds_dir"]) / f"{run_tag(cfg)}.csv")["val_auc"]
                chosen = {"selected": select_best(val_auc.tolist()), "last": int(cap["rounds"][-1])}
                for which in st["which"]:
                    r = chosen[which]
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
                            scores.setdefault((st["name"], model_name, which, attack, n), []).append(
                                (p, score, mem, r))

    rows = []
    for (setting, model_name, which, attack, n), runs in scores.items():
        member = runs[0][2]
        per_seed = [attack_auc(sc[mem], sc[~mem]) for _, sc, mem, _ in runs]
        mean_score = np.mean([sc for _, sc, _, _ in runs], axis=0)  # same patient order in every seed
        lo, hi = bootstrap_ci(mean_score[member], mean_score[~member])
        rows.append({"setting": setting, "model": model_name, "round": which, "attack": attack, "node": n,
                     "n_members": int(member.sum()), "n_nonmembers": int((~member).sum()),
                     "auc_mean_over_seeds": np.mean(per_seed), "auc_std_over_seeds": np.std(per_seed),
                     "auc_of_seed_mean_score": attack_auc(mean_score[member], mean_score[~member]),
                     "ci_low": lo, "ci_high": hi})
    for st in settings:  # all nodes together: concatenate the per-node, seed-averaged scores
        for model_name in models:
            for which in st["which"]:
                for attack in ATTACKS:
                    runs = [scores[(st["name"], model_name, which, attack, n)] for n in names]
                    s = np.concatenate([np.mean([sc for _, sc, _, _ in r], axis=0) for r in runs])
                    m = np.concatenate([r[0][2] for r in runs])
                    lo, hi = bootstrap_ci(s[m], s[~m])
                    rows.append({"setting": st["name"], "model": model_name, "round": which, "attack": attack,
                                 "node": "all", "n_members": int(m.sum()), "n_nonmembers": int((~m).sum()),
                                 "auc_of_seed_mean_score": attack_auc(s[m], s[~m]), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main():
    settings = [{"name": "adam_baseline", "cfg": {}, "capture_dir": OUT / "captures",
                 "rounds_dir": OUT / "rounds", "which": ("selected", "last")}]
    df = membership_table(settings)
    df.to_csv(OUT / "membership.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
