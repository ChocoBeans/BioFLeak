"""Gradient-inversion success against batch size (no defense): how the exact attack blurs as batches grow.

`python -m attacks.plot_inversion` -> results/figures/inversion_batch_size.png and inversion_batch_size_table.csv.
Needs results/attacks/inversion.csv (python -m attacks.run_inversion). Gradient matching is a lower bound on
leakage (fixed optimizer budget); the closed form exists only at batch size 1.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.preprocess import ROOT

FIG = ROOT / "results" / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"logreg": ("#2a78d6", "o", "Logistic regression"), "mlp": ("#eb6834", "s", "MLP")}  # palette slots 1, 2


def main():
    d = pd.read_csv(ROOT / "results" / "attacks" / "inversion.csv")
    gm = d[d.method == "gradient_matching"].groupby(["model", "batch_size"])["cosine"].mean().unstack(0)
    chance = d[d.method == "gradient_matching"].groupby("batch_size")["chance"].mean()
    closed = d[d.method == "closed_form"].groupby("model")["cosine"].mean()
    sizes = list(gm.index)
    pos = {b: i for i, b in enumerate(sizes)}

    plt.rcParams.update({"font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot([pos[b] for b in sizes], [chance[b] for b in sizes], color=MUTED, ls="--", lw=1.5, zorder=1)
    ax.text(1.5, -0.045, "chance (random patients)", color=MUTED, fontsize=11, ha="center", va="top")
    for m, (color, marker, label) in SERIES.items():
        ax.plot([pos[b] for b in sizes], gm[m], color=color, marker=marker, lw=2, ms=8, mec="white", mew=1.5,
                zorder=3)
        ax.annotate(label, (pos[8], gm[m].iloc[-1]), (pos[8] + 0.12, gm[m].iloc[-1]), color=INK, fontsize=12,
                    va="center")
        dx = -0.07 if m == "logreg" else 0.07  # side by side: both closed-form values are 1.00
        ax.scatter([pos[1] + dx], [closed[m]], s=130, color=color, marker=marker, edgecolor=INK, linewidth=1.6,
                   zorder=4)
    ax.annotate("closed form: exact\n(cosine 1.00, both models)", (pos[1], 1.0), (pos[1] + 0.3, 0.97),
                color=INK, fontsize=11, va="center")
    ax.set_xticks(range(len(sizes)), [str(b) for b in sizes])
    ax.set(xlabel="samples per client update (batch size)", ylabel="cosine similarity to the true patient",
           ylim=(-0.12, 1.08), xlim=(-0.3, len(sizes) - 0.1))
    ax.set_title("Exact at batch size 1, blurred as batches grow (no defense)", loc="left", color=INK, fontsize=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(0, -0.27, "Gradient matching (lines) is a lower bound: a fixed optimization budget, 3 seeds, rounds 1 / 10 / 50 / 100.",
            transform=ax.transAxes, color=MUTED, fontsize=9.5)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "inversion_batch_size.png", dpi=200)
    table = gm.rename(columns=lambda m: f"{m}_gradient_matching").assign(chance=chance)
    table.loc[1, [f"{m}_closed_form" for m in closed.index]] = closed.values
    table.round(3).to_csv(FIG / "inversion_batch_size_table.csv")
    print(table.round(3))


if __name__ == "__main__":
    main()
