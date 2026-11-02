"""Convergence plot: centralized (per epoch) vs FedAvg (per round).

`python -m utils.plot_convergence` reads results/centralized/epochs/ and results/federated/rounds/
(written by `python -m centralized.run` and `python -m federated.run`) and writes
results/figures/convergence.png plus the plotted numbers in results/figures/convergence_table.csv.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.preprocess import ROOT
from utils.train import SEEDS

RESULTS = ROOT / "results"
MODELS = {"logreg": "Logistic regression", "mlp": "MLP (100 → 32 → 1)"}
# The test row is zoomed on the plateau, where the final gap between the two settings is visible.
ROWS = [("val_auc", "Validation AUC", None), ("test_pooled_auc", "Test AUC (pooled)\naxis starts at 0.90", 0.90)]

# Categorical slots 1 and 2 of the reference palette (light surface); text stays in ink colors.
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#85847f", "#e4e3df"
SERIES = {"Centralized (per epoch)": "#2a78d6", "FedAvg (per round)": "#eb6834"}


def read_curves(kind, model):
    """(steps x seeds) frame of every logged column, stacked as a dict of frames."""
    frames = []
    for seed in SEEDS:
        if kind == "central":
            f = pd.read_csv(RESULTS / "centralized" / "epochs" / f"{model}_s{seed}.csv").set_index("epoch")
        else:
            f = pd.read_csv(RESULTS / "federated" / "rounds" / f"{model}_adam_e1_b32_s{seed}.csv").set_index("round")
        frames.append(f)
    return frames


def selected_steps(kind, model):
    df = pd.read_csv(RESULTS / ("centralized" if kind == "central" else "federated") / "metrics.csv")
    col = "best_epoch" if kind == "central" else "best_round"
    return df[(df.model == model) & (df.split == "pooled")][col].tolist()


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(len(ROWS), len(MODELS), figsize=(11, 7.6), sharex=True, sharey="row",
                             facecolor=SURFACE)
    table, lows = [], []
    for j, (model, model_name) in enumerate(MODELS.items()):
        for kind, label in zip(("central", "fed"), SERIES):
            frames, picks = read_curves(kind, model), selected_steps(kind, model)
            for i, (col, _, _) in enumerate(ROWS):
                ax = axes[i, j]
                m = pd.concat([f[col] for f in frames], axis=1)
                mean, std = m.mean(axis=1), m.std(axis=1)
                steps, color = mean.index.to_numpy(), SERIES[label]
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
                ax.plot(steps, mean, color=color, linewidth=2)
                pick = int(np.median(picks))  # median step chosen by early stopping on validation AUC
                ax.plot([pick], [mean.loc[pick]], "o", color=color, markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
                lows.append((mean - std).min())
                table += [{"model": model, "series": label, "metric": col, "step": s, "mean": a, "std": b}
                          for s, a, b in zip(steps, mean, std)]
    low = max(0.0, np.floor(min(lows) * 20) / 20)
    for i, (_, row_name, row_low) in enumerate(ROWS):
        for j, (_, model_name) in enumerate(MODELS.items()):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
            ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
            ax.minorticks_off()
            ax.set_ylim(low if row_low is None else row_low, 1.0)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0, labelsize=9)
            if i == 0:
                ax.set_title(model_name, loc="left", fontsize=11, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(row_name, fontsize=10)
            if i == len(ROWS) - 1:
                ax.set_xlabel("Training steps (epochs for centralized, rounds for FedAvg), log scale", fontsize=9)

    fig.suptitle("FedAvg learns a little slower and plateaus 0.01 to 0.02 AUC below centralized training",
                 x=0.06, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.06, 0.945, "Mean ± 1 std over 5 seeds. Dot = median step chosen by early stopping on validation AUC.",
             ha="left", fontsize=10, color=INK_2)
    handles = [plt.Line2D([0], [0], color=c, linewidth=2) for c in SERIES.values()]
    fig.legend(handles, list(SERIES), loc="upper left", bbox_to_anchor=(0.052, 0.925), frameon=False,
               ncol=2, fontsize=10, labelcolor=INK_2)
    fig.text(0.06, 0.012, "Validation AUC is pooled for centralized; for FedAvg it is the clients' per-node AUCs "
             "averaged by Flower (weighted by node size). Test AUC is on the same 140 held-out patients.",
             ha="left", fontsize=8, color=INK_3)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.89))

    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "convergence.png", dpi=200, facecolor=SURFACE)
    pd.DataFrame(table).to_csv(out / "convergence_table.csv", index=False)
    print(f"saved {out / 'convergence.png'} and {out / 'convergence_table.csv'}")


if __name__ == "__main__":
    main()
