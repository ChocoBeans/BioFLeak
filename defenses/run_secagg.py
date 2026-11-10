"""Simulated secure aggregation: the server learns only the sum of the client updates.

Pairwise additive masking (the core of Bonawitz et al.): for each client pair (i < j) a shared random
mask m_ij is added to client i's update and subtracted from client j's, so every mask cancels in the sum.
Real protocols add key agreement, dropout recovery and modular arithmetic; none of that changes what the
server sees, which is what is simulated here (float64, so the sum is exact to rounding error).

The server therefore has two things to attack, both tested on the batch-size-1 captures
(`python -m attacks.run_capture` first):
- one client's masked update: looks like noise, so the closed form recovers nothing;
- the aggregate sum (what FedAvg uses anyway): a weighted mix of the four clients' gradients, attacked with
  the closed form (a blend of the four samples) and with gradient matching over four samples with the
  FedAvg weights. Reconstructions are scored against the four true samples.
Secure aggregation leaves the global model unchanged, so utility is unchanged by construction; the check
below confirms the masks cancel.

Membership inference is tested on the SGD 50-round captures of the DP sweep (`python -m defenses.run_dp`,
epsilon = inf):
clients mask their FedAvg-weighted update (w_k * delta_k + mask_k, masks sum to zero), the server's
aggregate is exact, and the update-gain attack, which needs each client's local model, has only the masked
update to work with (the attacker divides by the known weight w_k and adds it to the global model). The
global-loss attack uses only the global model and should be unchanged; both are compared with the
unmasked run. Output: results/defenses/secagg_membership.csv.

`python -m defenses.run_secagg` -> results/defenses/secagg.csv, secagg_membership.csv and summary tables
(needs the batch-size-1 captures and the `defenses.run_dp` captures).
"""
import argparse

import numpy as np
import pandas as pd

from attacks.inversion import closed_form, gradient_matching, matched_cosine
from federated.server_app import DEFAULTS, check_run_config, run_tag
from utils.dataset import load_processed
from utils.preprocess import ROOT
from utils.standardization import standardize_local
from utils.train import batch_generator, sample_batches

OUT = ROOT / "results" / "defenses"
CAPTURES = ROOT / "results" / "attacks" / "captures"
MASK_STD = 1.0  # much larger than an update (about 1e-2), as masks in real protocols are uniform over a huge range


def mask_updates(deltas, rng):
    """Pairwise-masked copies of the (clients, params) updates; the masks cancel in the sum."""
    masked = deltas.copy()
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            m = rng.normal(0.0, MASK_STD, deltas.shape[1])
            masked[i] += m
            masked[j] -= m
    return masked


def masked_membership_captures(src, cfg, seeds=range(5), models=("logreg", "mlp")):
    """Copy the captures in `src` (runs described by `cfg`) with every client's update replaced by what
    the server would reconstruct from the masked, weighted one: (w_k * delta_k + mask_k) / w_k.
    Returns the folder of the masked copies."""
    nodes = standardize_local(*load_processed())[1]
    n = np.array([len(nodes[k]["train"][1]) for k in sorted(nodes)], dtype=float)
    w = n / n.sum()
    out = OUT / "captures_masked"
    out.mkdir(parents=True, exist_ok=True)
    rng, worst = np.random.default_rng(0), 0.0
    for model in models:
        for seed in seeds:
            tag = run_tag({**DEFAULTS, **cfg, "model": model, "seed": seed})
            cap = dict(np.load(src / f"{tag}.npz"))
            masked = np.stack([mask_updates(w[:, None] * d, rng) / w[:, None] for d in cap["delta"]])
            worst = max(worst, float(np.abs((w[None, :, None] * masked).sum(1)
                                            - (w[None, :, None] * cap["delta"]).sum(1)).max()))
            np.savez(out / f"{tag}.npz", **{**cap, "delta": masked})
    print(f"masked captures: largest |aggregate(masked) - aggregate(plain)| = {worst:.2e}")
    return out


def membership_under_secagg():
    """Update-gain and global-loss attacks, plain vs masked, on the SGD 50-round runs the DP results use
    (epsilon = inf)."""
    from attacks.run_membership import membership_table
    protocols = {
        "sgd": {"cfg": {"optimizer": "sgd", "learning-rate": 0.1, "num-server-rounds": 50, "dp-epsilon": 0.0},
                "src": OUT / "captures", "rounds_dir": OUT / "rounds", "which": ("last",)}}
    settings = []
    for name, p in protocols.items():
        masked = masked_membership_captures(p["src"], p["cfg"])
        base = {"cfg": p["cfg"], "rounds_dir": p["rounds_dir"], "which": p["which"]}
        settings += [{"name": f"{name}_plain", "capture_dir": p["src"], **base},
                     {"name": f"{name}_secagg", "capture_dir": masked, **base}]
    df = membership_table(settings)
    df.to_csv(OUT / "secagg_membership.csv", index=False)
    print(df[df.node == "all"].pivot_table(index=["model", "round", "attack"], columns="setting",
                                           values="auc_of_seed_mean_score").round(3))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["logreg", "mlp"])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, nargs="*", default=[1, 10, 50])
    parser.add_argument("--dlg-steps", type=int, default=500)
    args = parser.parse_args()

    nodes = standardize_local(*load_processed())[1]
    names = sorted(nodes)
    n_features = nodes[names[0]]["train"][0].shape[1]
    n_train = np.array([len(nodes[n]["train"][1]) for n in names], dtype=float)
    weights = n_train / n_train.sum()  # FedAvg weights
    rows, cancel_error = [], 0.0
    for model in args.models:
        for seed in args.seeds:
            cfg = {**DEFAULTS, "model": model, "seed": seed, "optimizer": "sgd", "local-steps": 1, "batch-size": 1}
            check_run_config(CAPTURES / f"{run_tag(cfg)}.npz", **{"learning-rate": 0.1})  # run_capture default
            cap = np.load(CAPTURES / f"{run_tag(cfg)}.npz")
            lr, rng = float(cap["lr"]), np.random.default_rng(seed)
            for rnd in args.rounds:
                i = int(np.where(cap["rounds"] == rnd)[0][0])
                deltas = cap["delta"][i]
                masked = mask_updates(deltas, rng)
                cancel_error = max(cancel_error, float(np.abs(masked.sum(0) - deltas.sum(0)).max()))
                truth = np.vstack([nodes[n]["train"][0][sample_batches(
                    len(nodes[n]["train"][1]), 1, 1, batch_generator(seed, rnd, c))[0].numpy()]
                    for c, n in enumerate(names)])  # one true sample per client
                base = {"model": model, "seed": seed, "round": rnd}

                # 1) one client's masked update, closed form (the attacker would have recovered the sample)
                for c in range(len(names)):
                    rec = closed_form(model, n_features, -masked[c] / lr)
                    rows.append({**base, "target": "masked_update", "method": "closed_form",
                                 "cosine": matched_cosine(rec, truth[c])})
                    rows.append({**base, "target": "plain_update", "method": "closed_form",
                                 "cosine": matched_cosine(closed_form(model, n_features, -deltas[c] / lr), truth[c])})

                # 2) the aggregate (all the server legitimately gets)
                agg = -(weights[:, None] * deltas).sum(0) / lr
                blend = closed_form(model, n_features, agg)
                rows.append({**base, "target": "aggregate", "method": "closed_form_blend",
                             "cosine": float(np.mean([matched_cosine(blend, t) for t in truth]))})
                rec = gradient_matching(model, n_features, cap["glob"][i], agg, len(names),
                                        steps=args.dlg_steps, seed=seed, weights=weights)
                rows.append({**base, "target": "aggregate", "method": "gradient_matching",
                             "cosine": matched_cosine(rec, truth)})
        print(f"{model} done", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "secagg.csv", index=False)
    print(f"largest |sum(masked) - sum(plain)| over all runs: {cancel_error:.2e}")
    print(df.groupby(["model", "target", "method"])["cosine"].agg(["mean", "std", "count"]).round(3))
    membership_under_secagg()


if __name__ == "__main__":
    main()
