# FLeak

*2026-10-08*

Mini-project proposal: federated learning on Chronic Lymphocytic Leukemia (CLL, Broad, Nat Genet 2022) data via cBioPortal, testing whether FL matches centralized performance and how much raw patient data leaks from FL model updates.

## Motivation

Federated learning (FL) is increasingly proposed as a way to let multiple institutions collaboratively train models on sensitive biomedical data (genomic, clinical, imaging) without centralizing raw patient records. Published results show real gains from this approach: COVID-19 outcome prediction improved 16–38% at 24h/72h, and rare tumour segmentation improved substantially under FL versus single-site training.

However, FL is frequently described as "privacy-preserving" without that claim being rigorously tested. Reviews of FL frameworks in biomedical research note scarce integration of privacy-preserving techniques (differential privacy, secure aggregation) and a predominant reliance on simpler, less protected architectures — meaning raw data may still be at risk of leaking through shared model updates via gradient inversion or membership inference attacks.

This project has two goals:
1. Demonstrate that FL can match centralized model performance on a real cancer classification task across simulated institutional silos.
2. Measure how much raw patient data actually leaks from vanilla FL updates, then apply and evaluate defenses (differential privacy, secure aggregation) to close that gap — reporting the resulting privacy-utility tradeoff.

## Dataset & goal

**Dataset:** cBioPortal — Chronic Lymphocytic Leukemia (Broad, Nature Genetics 2022), study ID `cll_broad_2022`. The study has 1,154 tumour samples, but RNA-seq exists for only a subset; 700 samples (from 696 patients) have both RNA-seq and an IGHV label and are used here.

**Task:** Binary classification — IGHV-mutated (M-CLL) vs. IGHV-unmutated (U-CLL) CLL, a lab-measured marker that separates two clinically distinct forms of the disease with different prognosis and treatment. Input is gene expression (RNA-seq, TPM). Classes are roughly balanced (51% mutated).

**Features:** The 100 highest-variance genes, log2(TPM+1) transformed, after excluding immunoglobulin genes (a near-proxy for the label and clone-specific fingerprints), Y-chromosome/sex-linked genes, pseudogenes, small RNAs and other noisy transcripts (classified with the GENCODE v19 annotation). Date-mangled gene symbols from the source file (e.g. `10-Sep`) are restored (`SEPT10`). No curated panel is assumed.

**Node split:** Samples are grouped by contributing cohort (`COHORT`: ICGC, GCLLSG, DFCI, NHLBI, UCSD, MDACC), pooling the three cohorts with under 40 patients (NHLBI, UCSD, MDACC) into "OTHER". This simulates different institutions as different nodes. This gives 4 nodes (n = 289 / 229 / 130 / 52 samples) whose label rates differ (34%–63% mutated) — the non-IID condition FL needs to prove itself against.

**Train/val/test:** Splits are made at the patient level (a patient's samples never straddle splits). A shared test set (20% of patients, stratified by cohort and label) is held out first and used for both centralized and federated models; the rest is split within each node into train/val (80/20). With seed 4171 the pooled sizes are 447 train / 113 val / 140 test samples (51–52% mutated in each); per-node sizes are in `DATA_NOTES.txt`.

**Preprocessing:** Features are z-scored using training data only — pooled statistics for the centralized model, each node's own statistics for federated learning. Selection and preprocessing choices are documented in `DATA_NOTES.txt`.

## Why CLL over other cancer types

- Recent (2022), large, and open, with RNA-seq and a clean binary clinical label — no survival analysis or censored data.
- The label is lab-measured, not derived from the expression data, so the task is not circular.
- Multiple contributing cohorts give a natural multi-institution split with heterogeneous label rates.
- Limitation: only four usable nodes after filtering, and no direct FL benchmark from prior work for this dataset.

## Part 1: Centralized vs. Federated

| Item | Choice |
| --- | --- |
| Train | Classifier (logistic regression or small MLP) on the top-variance gene set |
| Label | IGHV mutation status (binary) |
| Nodes | Split by contributing cohort (`COHORT`) |
| Metrics | AUC-ROC (primary), plus accuracy, F1, precision, recall; reported pooled and per node, as mean ± std over seeds |
| Comparison | Centralized model vs. FedAvg (Flower) |

Keep the model simple (logistic regression / small MLP) so the threat-model attacks in Part 2 are easier to implement and interpret.

## Part 2: Threat model

**Attacks:**
- Gradient inversion attack — reconstruct approximate patient feature vectors from a client's shared gradient update.
- Membership inference attack — determine whether a given patient's record was in a specific site's training set.

**Attack success metrics:**
- Reconstruction similarity (e.g., cosine similarity between reconstructed and real feature vectors) for inversion.
- AUC for membership inference.

**Defenses:**
- Differential privacy via Opacus — add calibrated noise to gradients, sweep privacy budgets (ε = 1, 3, 8, ∞).
- Secure aggregation — simulated via Flower's support or a simplified masking scheme, so the server only sees summed updates, never individual client gradients.

**Core result:** Privacy-utility tradeoff curve (AUC-ROC vs. ε) and leakage-reduction curve (attack success vs. ε). Look for a "knee" in the curve — a privacy budget where attack success drops sharply while accuracy cost stays small.

## Next steps / open questions

- [x] Decide cBioPortal access route: bulk download (`https://datahub.assets.cbioportal.org/cll_broad_2022.tar.gz`), fetched automatically by `utils/preprocess.py`
- [x] Preprocess data into per-node train/val splits and a shared test set (`python -m utils.preprocess`)
- [ ] Confirm cohort split has enough patients per node for stable training (4 nodes; the smallest, OTHER, has 34 train samples (30 patients), 8 val and 10 test)
- [ ] Pick final feature set size (top-N variance genes; try N = 50 / 100 / 500)
- [x] FedAvg baseline with Flower (`python -m federated.run`)
- [ ] Per-client update access on the server for the attacks (SGD, 1 local step, small batches)
- [ ] Frame for week 7 proposal: two research questions — (1) does FL match centralized performance for cancer classification across institutional silos, and (2) how much raw patient data leaks from FL updates, and what's the cost of closing that leak

## Project layout

```
FLeak/
├── data/        # raw, reference and processed CLL data (gitignored)
│   ├── cll_broad_2022/   # raw cBioPortal download
│   ├── reference/        # GENCODE v19 gene annotation (downloaded on first run)
│   └── processed/        # samples_joined.csv, test.csv, nodes/<node>/{train,val}.csv
├── utils/
│   ├── preprocess.py     # download/load raw files, join/filter, gene selection, patient-level splits
│   ├── standardization.py  # pooled (centralized) and local (federated) z-scoring from training data
│   ├── train.py          # shared training epoch and evaluation (AUC, accuracy, F1, precision, recall)
│   └── dataset.py        # reads data/processed/ into arrays (load_processed)
├── models/
│   ├── logistic_regression.py  # logistic regression (one linear layer, outputs a logit)
│   ├── mlp.py            # small MLP (100 -> 32 -> 1, outputs a logit)
│   └── __init__.py       # build_model("logreg" | "mlp", n_features)
├── centralized/
│   └── run.py            # centralized baseline: pooled training, per-epoch curves, early-stopping selection, 5 seeds
├── federated/
│   ├── client_app.py     # Flower client: one per node, local training/validation
│   ├── server_app.py     # FedAvg, client-side validation + server-side test each round
│   ├── run.py            # driver: models x seeds, round selection, results/federated/metrics.csv
│   ├── custom_strategy.py  # imported Flower example, kept for reference only (unused)
│   └── pyproject.toml, LICENSE  # dependency list from the Flower example
├── attacks/     # gradient inversion and membership inference attack implementations
├── results/     # metrics output (CSV, gitignored); centralized/ and federated/ subfolders
├── notebooks/   # exploratory analysis
├── DATA_NOTES.txt  # data selection and preprocessing decisions, with reasoning
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.11.4 or newer (developed on 3.12).

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Preparing the data

```bash
python -m utils.preprocess   # run from the repo root
```

This downloads and extracts the CLL study from cBioPortal into `data/cll_broad_2022/` if it isn't there yet (about 136 MB), joins/filters the tables, selects genes, splits into nodes, and writes `data/processed/`. The first run also downloads the GENCODE v19 gene annotation (about 38 MB) into `data/reference/`, so it needs internet. Output:

- `data/processed/samples_joined.csv` — the 700 joined sample/patient rows with RNA-seq and an IGHV label.
- `data/processed/nodes/<node>/train.csv` and `val.csv` — one folder per FL node (DFCI, GCLLSG, ICGC, OTHER).
- `data/processed/test.csv` — the shared held-out test set, with a `NODE` column.

Each split file has `SAMPLE_ID`, `PATIENT_ID`, `COHORT`, `NODE`, `label` (1 = IGHV-mutated) and the unscaled log2(TPM+1) values of the 100 selected genes. Standardization is applied at load time (`standardize_pooled` for the centralized model, `standardize_local` for federated learning). The seed is fixed at 4171 (`SEED` in `utils/preprocess.py`). See `DATA_NOTES.txt` for the reasoning behind each choice.

## Using the data in code

```python
from utils.preprocess import load_cll, make_splits
from utils.standardization import standardize_pooled, standardize_local

X, y, node, patient, genes = load_cll()                # unscaled log2(TPM+1), 100 genes
test, nodes = make_splits(X, y, node, patient)         # same split as data/processed/
test, nodes = standardize_pooled(test, nodes)          # centralized; or standardize_local for FL
# nodes["ICGC"]["train"] -> (X, y); test -> (X, y, node_per_sample)
```

`utils.dataset.load_processed()` reads the CSVs in `data/processed/` into the same `(test, nodes)` layout (unscaled); it gives the same arrays as `load_cll` + `make_splits` (seed 4171). Use it with `standardize_pooled` / `standardize_local`.

## Running the centralized baseline

```bash
python -m centralized.run   # run from the repo root
```

Trains logistic regression (`logreg`) and the small MLP (100 → 32 → 1) on the pooled node training sets (pooled standardization), trains for 200 epochs while logging the pooled validation metrics and the test metrics (pooled and per node) at every epoch, and then picks the epoch by early stopping on the pooled validation AUC (patience 20, `utils.train.select_best`, the same rule the federated run uses), never from test numbers, over 5 seeds. Seeds vary model initialisation and batch order; the data split is fixed. Hyperparameters (Adam, learning rate 1e-3, weight decay 1e-4, batch size 32) live in `utils/train.py` so the federated run can use the same ones. Results are written to `results/centralized/metrics.csv`, and the per-epoch curves to `results/centralized/epochs/<model>_s<seed>.csv`. These have the same columns as the federated per-round files in `results/federated/rounds/` (`epoch` instead of `round`), so convergence can be plotted side by side, in epochs for centralized and in rounds for federated. The two units are not equivalent: a round averages four separately trained local models.

Results (pooled test set of 140 samples, mean ± std over 5 seeds):

| Model | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic regression | 0.975 ± 0.004 | 0.903 ± 0.015 | 0.904 ± 0.016 | 0.915 ± 0.016 | 0.894 ± 0.030 |
| MLP | 0.977 ± 0.002 | 0.920 ± 0.013 | 0.921 ± 0.014 | 0.934 ± 0.007 | 0.908 ± 0.027 |

Per-node test AUC (logreg / MLP): DFCI 0.996 / 0.996, GCLLSG 0.910 / 0.920, ICGC 0.997 / 0.998, OTHER 0.914 / 1.000 (OTHER has only 10 test samples).

## Running the federated baseline

```bash
source venv/bin/activate     # Flower's simulation needs the venv on PATH
python -m federated.run      # run from the repo root
```

Simulates FedAvg across the four nodes with Flower (`flwr.simulation.run_simulation`, in-process; `flwr run` is not used because it builds a separate environment for every run). Same models, Adam settings (learning rate 1e-3, weight decay 1e-4, batch size 32) and seeds as the centralized run, 1 local epoch per round, at most 200 rounds, and each client standardizes with its own training statistics. Validation stays at the clients: every round Flower averages the clients' validation metrics (weighted by node size), and the round is chosen by early stopping (patience 20) on that average validation AUC, never on test numbers. The shared test set is held by the server for evaluation only and scored each round, pooled and per node. Per-round curves are in `results/federated/rounds/`; the summary is in `results/federated/metrics.csv` (same columns as the centralized file, plus `best_round`, optimizer, local epochs and batch size). Options: `--optimizer adam|sgd`, `--local-epochs`, `--batch-size`, `--rounds`, `--seeds`, `--models`. A full run takes about 15 to 20 minutes.

Pooled test set (140 samples, mean ± std over 5 seeds):

| Model | Setting | AUC | Accuracy | F1 |
|---|---|---|---|---|
| Logistic regression | centralized | 0.975 ± 0.004 | 0.903 ± 0.015 | 0.904 ± 0.016 |
| Logistic regression | FedAvg | 0.954 ± 0.010 | 0.909 ± 0.037 | 0.912 ± 0.037 |
| MLP | centralized | 0.977 ± 0.002 | 0.920 ± 0.013 | 0.921 ± 0.014 |
| MLP | FedAvg | 0.963 ± 0.005 | 0.921 ± 0.020 | 0.925 ± 0.019 |

Per-node AUC is nearly the same as centralized (DFCI ≈ 0.99, ICGC ≈ 0.998, GCLLSG ≈ 0.91, OTHER has only 10 test samples). Most of the lower pooled AUC comes from local standardization, not from federation: the centralized models trained with local scaling get 0.965 (logistic regression) and 0.960 (MLP). Scores from differently scaled nodes are less comparable across nodes, which hurts the pooled AUC but not the per-node AUC.

## Convergence plot

```bash
python -m utils.plot_convergence   # needs results/centralized/epochs/ and results/federated/rounds/
```

Writes `results/figures/convergence.png` (validation and test AUC against epochs for centralized and rounds for FedAvg, mean ± 1 std over 5 seeds, log-scale x-axis, test row zoomed to 0.90–1.0) and the plotted numbers in `results/figures/convergence_table.csv`. FedAvg needs more steps to learn (logistic regression reaches test AUC 0.95 after about 17 rounds versus 6 epochs centralized) and plateaus about 0.01 to 0.02 AUC lower. Validation AUC is not defined the same way in the two settings: pooled for centralized, the node-size-weighted mean of per-node AUCs for FedAvg.

## Research referenced

- [Efficacy of federated learning on genomic data: a study on the UK Biobank and the 1000 Genomes Project](https://pmc.ncbi.nlm.nih.gov/articles/PMC10937521/) — FL applicability for phenotype and ancestry prediction; shows federated models approach centralized performance despite inter-node heterogeneity.
- [Federated learning frameworks: quality and interoperability for biomedical research](https://pmc.ncbi.nlm.nih.gov/articles/PMC12862364/) — review noting scarce integration of privacy-preserving techniques and reliance on simpler architectures in current FL frameworks.
- [Federated Learning for the pathogenicity annotation of genetic variants](https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaf523/64325337/btaf523.pdf) — multi-institutional FL vs. centralized comparison for coding SNVs, non-coding SNVs, and CNVs; found larger centralized-vs-FL divergence for CNVs specifically.
- [Technical and Legal Aspects of Federated Learning in Bioinformatics: Applications, Challenges and Opportunities](https://arxiv.org/pdf/2503.09649) — documents FL's measured gains in COVID-19 outcome prediction and rare tumour segmentation.
- [Eleven quick tips for Biomedical Federated Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC13496424/) — practical guidance on setting up biomedical FL projects, including use of public data for initial test scenarios.
- [Federated Learning for Histopathology Image Classification: A Systematic Review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12785327/) — PRISMA review of 24 FL studies in histopathology (2020–2025), showing steady growth in that subfield.
- [pfl-research: simulation framework for accelerating research in Private Federated Learning](https://arxiv.org/pdf/2404.06430) — reference for FL simulation frameworks and private FL research tooling.
- [Real-World Image Datasets for Federated Learning](https://arxiv.org/pdf/1910.11089) — background on non-IID data partitioning for FL benchmarks.
