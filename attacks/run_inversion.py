"""Score the gradient-inversion attack on the captured updates (`python -m attacks.run_capture` first).

For every captured client update it reconstructs the batch and compares with the true batch, which
is recomputed from the seeded generators (the attacker never sees it). Metric: mean cosine similarity
between reconstructed and true feature vectors (locally standardized space). Reference ("chance")
level: cosine between the true batch and randomly drawn other training samples of the same node.

`python -m attacks.run_inversion` -> results/attacks/inversion.csv and a summary table.
"""
import argparse

import numpy as np
import pandas as pd

from attacks.inversion import closed_form, gradient_matching, infer_label, matched_cosine
from federated.server_app import run_tag
from utils.dataset import load_processed
from utils.preprocess import ROOT
from utils.standardization import standardize_local
from utils.train import batch_generator, sample_batches

OUT = ROOT / "results" / "attacks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["logreg", "mlp"])
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--dlg-rounds", type=int, nargs="*", default=[1, 10, 50, 100],
                        help="rounds attacked with gradient matching (the closed form uses every round)")
    parser.add_argument("--dlg-steps", type=int, default=500)
    args = parser.parse_args()

    nodes = standardize_local(*load_processed())[1]
    names = sorted(nodes)
    n_features = nodes[names[0]]["train"][0].shape[1]
    rows = []
    for model in args.models:
        for batch in args.batch_sizes:
            for seed in args.seeds:
                cfg = {"model": model, "seed": seed, "optimizer": "sgd", "local-epochs": 1,
                       "local-steps": 1, "batch-size": batch}
                cap = np.load(OUT / "captures" / f"{run_tag(cfg)}.npz")
                lr = float(cap["lr"])
                rng = np.random.default_rng(seed)
                for i, rnd in enumerate(cap["rounds"]):
                    for c, node in enumerate(names):
                        X = nodes[node]["train"][0]
                        idx = sample_batches(len(X), 1, batch, batch_generator(seed, int(rnd), c))[0].numpy()
                        truth, grad = X[idx], -cap["delta"][i, c] / lr
                        other = rng.choice(np.setdiff1d(np.arange(len(X)), idx), size=batch, replace=False)
                        base = {"model": model, "batch_size": batch, "seed": seed, "round": int(rnd),
                                "node": node, "chance": matched_cosine(X[other], truth)}
                        if batch == 1:
                            rec = closed_form(model, n_features, grad)
                            rows.append({**base, "method": "closed_form", "cosine": matched_cosine(rec, truth)})
                        if int(rnd) in args.dlg_rounds:
                            rec = gradient_matching(model, n_features, cap["glob"][i], grad, batch,
                                                    steps=args.dlg_steps, seed=seed,
                                                    label=infer_label(grad) if batch == 1 else None)
                            rows.append({**base, "method": "gradient_matching",
                                         "cosine": matched_cosine(rec, truth)})
                print(f"{model} B={batch} seed={seed} done", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "inversion.csv", index=False)
    print(df.groupby(["model", "method", "batch_size"])[["cosine", "chance"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
