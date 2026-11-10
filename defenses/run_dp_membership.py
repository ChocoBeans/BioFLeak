"""Membership inference against the DP runs (`python -m defenses.run_dp` first, it saves the captures).

Same attack and metric as attacks/run_membership.py, on the model after the last of the 50 rounds (no
early stopping: selecting a round on validation data would spend extra privacy budget). epsilon = inf is
the non-private SGD run with the same optimizer, learning rate and rounds.

`python -m defenses.run_dp_membership` -> results/defenses/dp_membership.csv and a summary table.
"""
import pandas as pd

from attacks.run_membership import membership_table
from defenses.run_dp import OUT

EPSILONS = (1.0, 3.0, 8.0, float("inf"))


def main():
    settings = [{"name": f"eps={e:g}", "cfg": {"optimizer": "sgd", "num-server-rounds": 50,
                                               "dp-epsilon": 0.0 if e == float("inf") else e},
                 "capture_dir": OUT / "captures", "rounds_dir": OUT / "rounds", "which": ("last",)}
                for e in EPSILONS]
    df = membership_table(settings)
    df.to_csv(OUT / "dp_membership.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[df.node == "all"].round(3).to_string(index=False))
    print(df[df.node != "all"].pivot_table(index=["model", "attack", "node"], columns="setting",
                                          values="auc_of_seed_mean_score").round(3))


if __name__ == "__main__":
    main()
