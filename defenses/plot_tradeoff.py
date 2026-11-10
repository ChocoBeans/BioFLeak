"""Privacy-utility and leakage-reduction curves against epsilon (epsilon = inf drawn at the right edge).

`python -m defenses.plot_tradeoff` -> results/figures/privacy_utility.png and privacy_utility_table.csv.
Needs results/defenses/dp_utility.csv, dp_membership.csv and results/attacks/inversion_dp.csv.
Note the inversion runs (batch size 1, 1 step per round) and the utility / membership runs (batch 32, a
pass over the data per round) calibrate the noise for different step counts and sampling rates, so the
same epsilon means the same record-level guarantee per site, not the same noise per step.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.preprocess import ROOT

D, A, FIG = ROOT / "results" / "defenses", ROOT / "results" / "attacks", ROOT / "results" / "figures"
X = {1.0: 0, 3.0: 1, 8.0: 2, float("inf"): 3}  # evenly spaced positions
LABELS = ["1", "3", "8", "inf"]
COLOR = {"logreg": "#1f77b4", "mlp": "#d95f02"}


def main():
    util = pd.read_csv(D / "dp_utility.csv").query("split == 'pooled'")
    util = util.groupby(["model", "epsilon"])["auc"].agg(["mean", "std"]).reset_index()
    mem = pd.read_csv(D / "dp_membership.csv").query("node == 'all' and attack == 'update_gain'")
    mem["epsilon"] = mem["setting"].str.replace("eps=", "").astype(float)
    inv = pd.read_csv(A / "inversion_dp.csv").query("method == 'closed_form' and `round` <= 50")
    inv["epsilon"] = inv["epsilon"].replace(0.0, float("inf"))
    inv = inv.groupby(["model", "epsilon"])["cosine"].mean().reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in ("logreg", "mlp"):
        u, i, b = (t[t.model == m].sort_values("epsilon") for t in (util, inv, mem))
        ax[0].errorbar([X[e] for e in u.epsilon], u["mean"], u["std"], marker="o", color=COLOR[m], label=m)
        ax[1].plot([X[e] for e in i.epsilon], i["cosine"], marker="o", color=COLOR[m], label=m)
        ax[2].errorbar([X[e] for e in b.epsilon], b["auc_of_seed_mean_score"],
                       [b["auc_of_seed_mean_score"] - b.ci_low, b.ci_high - b["auc_of_seed_mean_score"]],
                       marker="o", color=COLOR[m], label=m)
    ax[0].set(title="Utility: pooled test AUC", ylabel="AUC")
    ax[1].set(title="Gradient inversion (batch 1, closed form)", ylabel="cosine similarity to true sample")
    ax[2].set(title="Membership inference (update gain)", ylabel="attack AUC (0.5 = guessing)")
    ax[2].axhline(0.5, color="grey", lw=0.8, ls="--")
    for a in ax:
        a.set_xticks(range(4), LABELS)
        a.set_xlabel("epsilon per site (delta = 1e-5)")
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "privacy_utility.png", dpi=150)

    # slide version: two panels, large type (membership is near chance everywhere and goes in the slide text)
    plt.rcParams.update({"font.size": 17})
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.6))
    for m in ("logreg", "mlp"):
        u, i = (t[t.model == m].sort_values("epsilon") for t in (util, inv))
        label = {"logreg": "Logistic regression", "mlp": "MLP"}[m]
        ax2[0].errorbar([X[e] for e in u.epsilon], u["mean"], u["std"], marker="o" if m == "logreg" else "s",
                        color=COLOR[m], lw=2.5, ms=9, capsize=4, label=label)
        ax2[1].plot([X[e] for e in i.epsilon], i["cosine"], marker="o" if m == "logreg" else "s",
                    color=COLOR[m], lw=2.5, ms=9)
    ax2[0].set(title="Utility: test AUC", ylabel="test AUC")
    ax2[1].set(title="Exact attack (cosine)", ylabel="cosine similarity")
    ax2[0].legend(loc="lower right", fontsize=15, frameon=False)
    for a in ax2:
        a.set_xticks(range(4), ["1", "3", "8", "\u221e"])
        a.set_xlabel("\u03b5 (smaller = more private)")
        a.grid(alpha=0.3)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(FIG / "privacy_utility_slides.png", dpi=200)
    table = (util.rename(columns={"mean": "test_auc", "std": "test_auc_std"})
             .merge(inv.rename(columns={"cosine": "inversion_cosine"}), on=["model", "epsilon"])
             .merge(mem[["model", "epsilon", "auc_of_seed_mean_score"]]
                    .rename(columns={"auc_of_seed_mean_score": "membership_auc"}), on=["model", "epsilon"]))
    table.round(3).to_csv(FIG / "privacy_utility_table.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
