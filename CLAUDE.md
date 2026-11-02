# BioFLeak

Mini-project: federated learning (FL) on CLL RNA-seq data (cBioPortal `cll_broad_2022`) to test
(1) whether FL matches centralized performance and (2) how much raw patient data leaks from FL
updates (gradient inversion, membership inference) and what defenses (DP, secure aggregation) cost.

- `README.md` — project plan, task definition, threat model, setup and run instructions.
- `DATA_NOTES.txt` — every data selection/preprocessing decision with its reasoning and the
  limitations. Read it before changing anything in `utils/preprocess.py`; update it when a data
  decision changes.

## Commands

```bash
source venv/bin/activate
pip install -r requirements.txt
python -m utils.preprocess      # run from the repo root; downloads data on first run, writes data/processed/
```

Python 3.11.4+ (developed and tested on 3.12.3). First run needs internet (study archive ~136 MB,
GENCODE v19 annotation ~38 MB into `data/reference/`). `data/` and `results/` are gitignored.

### Setting up on another machine (clone, then rebuild everything that is gitignored)

```bash
git clone https://github.com/ChocoBeans/BioFLeak.git && cd BioFLeak
python3 -m venv venv && source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt                         # large: torch, ray, jupyter
python -m utils.preprocess        # ~170 MB downloads, a few minutes; recreates data/
python -m centralized.run         # ~1.5 min on 16 cores; recreates results/centralized/
python -m federated.run           # ~15-20 min; recreates results/federated/ (venv must be active)
python -m utils.plot_convergence  # results/figures/convergence.png
```

- Tested versions: Python 3.12.3, flwr 1.39.0 (pinned `>=1.39.0,<1.40`), torch 2.14.1, ray 2.55.1,
  numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, opacus 1.6.0. Newer Flower (a fresh resolve gives
  1.42) was not tested.
- **Use Linux, macOS or WSL2.** Ray (Flower's simulation backend) is only experimental on native
  Windows. All development happened in WSL2 (Ubuntu); nothing is Windows-tested.
- The venv must be active (or `venv/bin` on PATH) when running `federated.run`; Flower looks up its
  helper commands on PATH.
- Code is CPU-only (tensors are never moved to a GPU), so a GPU PC changes nothing. The CUDA
  "driver too old" warning seen in WSL is harmless.
- Results regenerated on another machine can differ in the last digits (thread count and CPU
  floating point), not in the conclusions. The data split is deterministic (`SEED = 4171`).
- Unpushed commits and untracked files do not travel with `git clone`: push first, and check
  `git status`.

## Layout

- `utils/preprocess.py` — download, join/filter, gene selection, splits, CSV export (`python -m utils.preprocess`).
- `utils/standardization.py` — `standardize_pooled` / `standardize_local`.
- `utils/train.py` — shared `train_epoch`, `evaluate` (AUC, accuracy, F1, precision, recall).
- `utils/dataset.py` — `load_processed()` reads `data/processed/` into `(test, nodes)` (unscaled).
- `models/logistic_regression.py`, `models/mlp.py` — the two models (raw-logit output, train with
  `BCEWithLogitsLoss`); `models.build_model("logreg" | "mlp", n_features)` builds either. Logistic
  regression is the main model; the MLP is the comparison.
- `centralized/run.py` — centralized baseline (`python -m centralized.run`), writes `results/centralized/metrics.csv` and per-epoch curves in
  `results/centralized/epochs/` (same columns as `results/federated/rounds/`, `epoch` vs `round`). Trains all
  200 epochs, then picks the epoch with `utils.train.select_best` (early stopping, patience 20, shared with
  the federated run). Shared hyperparameters (`LR`, `WEIGHT_DECAY`, `BATCH_SIZE`, `SEEDS`, `PATIENCE`) are in
  `utils/train.py`.
- `federated/` — FedAvg with Flower (`python -m federated.run`, writes `results/federated/metrics.csv`
  and per-round curves in `results/federated/rounds/`):
  - `client_app.py` — one client per node (partition-id -> sorted node names); trains and validates on
    its own node only, local standardization; hyperparameters arrive in the per-round config message.
  - `server_app.py` — FedAvg; logs the client-side validation metrics (Flower's weighted average) and
    a server-side test evaluation (pooled and per node) each round; `make_server_app(cfg)`.
  - `run.py` — driver: models x seeds, picks the round from client-side validation AUC (patience 20,
    same as centralized), never from test numbers. Options: `--optimizer adam|sgd`, `--local-epochs`,
    `--batch-size` (attack runs: sgd, 1 epoch, small batches).
  - `custom_strategy.py` — kept for reference only (imported Flower example, imports `wandb`, unused).
  - `pyproject.toml` / `LICENSE` — dependency list from the Flower example (Apache-2.0); keep in sync
    with `requirements.txt`. `flwr run` is NOT used: it builds a private environment per run
    (minutes each) and its copy of the app cannot import `utils/` or `models/`. The simulation uses
    `flwr.simulation.run_simulation` (deprecated in Flower 1.39 but works) with config sent in messages.
  - Needs `ray` (in `requirements.txt` via `flwr[simulation]`). Run from the repo root with the venv active.
    Flower is pinned to the tested line (`>=1.39.0,<1.40`): an unpinned fresh install pulls 1.42, whose
    API was not tested here. Re-test the federated run before widening the range.
- `utils/plot_convergence.py` — `python -m utils.plot_convergence` -> `results/figures/convergence.png` (+ table CSV):
  centralized epochs vs FedAvg rounds, validation and test AUC, from the per-epoch / per-round files.
- `attacks/`, `notebooks/` — empty so far. Results go to `results/centralized/` and `results/federated/`.
- `data/` — raw study, GENCODE reference and `processed/` outputs (all gitignored).

Dependency direction: `utils` <- `models` <- `centralized`, `federated`, `attacks`. `centralized` and
`federated` must not import each other; both use the same `utils.train` and the same hyperparameters.
Run modules from the repo root with the venv active (`python -m centralized.run`, `python -m federated.run`).

In code: `from utils.preprocess import load_cll, make_splits` and
`from utils.standardization import standardize_pooled, standardize_local`. `utils.dataset.load_processed()`
reads the `data/processed/` CSVs and gives the same split as `load_cll` + `make_splits`.

## Status

### Done
- Dataset chosen: CLL (Broad, Nat Genet 2022). Task: IGHV mutated (1) vs unmutated (0) from gene
  expression. Replaced the original TCGA-BRCA plan (no ER/HER2 labels in that study).
- `utils/preprocess.py`: auto-download, join sample + patient tables, filter to samples in the
  expression matrix and with an IGHV label (700 samples, 696 patients), gene filtering with
  GENCODE v19 (drops chrY/XIST, pseudogenes, Ig loci, small RNAs, noisy transcripts), restores
  date-mangled symbols (`10-Sep` -> `SEPT10`), top-100 variance genes, log2(TPM+1).
- Nodes = contributing cohort (ICGC, GCLLSG, DFCI, OTHER); patient-level train/val per node and a
  shared 20% test set, seed 4171. Outputs in `data/processed/` (`nodes/<node>/{train,val}.csv`,
  `test.csv`, `samples_joined.csv`) were verified against the raw files.
- `standardize_pooled` (centralized) and `standardize_local` (federated) implemented in
  `utils/standardization.py`; applied at load time, not stored in the CSVs (which hold unscaled
  log2(TPM+1) values).
- Models written in PyTorch (so Flower, Opacus and the attacks can reuse them): logistic regression
  and a small MLP, one file each in `models/`. An earlier single-file attempt was removed.
- Centralized baseline done (`centralized/run.py`, 5 seeds, epoch picked by early stopping on val AUC, per-epoch curves logged). Pooled test AUC:
  logreg 0.975 ± 0.004, MLP 0.977 ± 0.002; accuracy 0.903 / 0.920. Weakest node: GCLLSG (AUC ~0.91-0.92);
  OTHER has only 10 test samples. These are the numbers FedAvg has to be compared against.
- Federated baseline done (`federated/`, FedAvg, 4 nodes, Adam, 1 local epoch, same hyperparameters and
  seeds as centralized). Pooled test AUC: logreg 0.954 ± 0.010, MLP 0.963 ± 0.005 (centralized 0.975 /
  0.977); per-node AUCs match centralized closely. Most of the pooled gap is local standardization, not
  federation: centralized trained with local scaling gets logreg 0.965 / MLP 0.960.
- Convergence plot done (`utils/plot_convergence.py`): FedAvg learns more slowly (logreg reaches test AUC
  0.95 after ~17 rounds vs ~6 epochs) and plateaus 0.01-0.02 below centralized.
- Project scaffold created: `models/`, `centralized/`, `federated/` (renamed from `fl/`), plus
  `utils/train.py` and `utils/dataset.py` (shared training/evaluation and the CSV reader, smoke-tested). README and DATA_NOTES are up to date with the above.

### To do (next: attacks)
Settled design, from the discussion so far:
- **Main model: logistic regression; MLP as the comparison** (leakage vs capacity). FedAvg is the
  method; the attacked runs use the same FedAvg code with `--optimizer sgd --local-epochs 1` and a
  small batch (1-8) so each client update is essentially one gradient (`-lr * gradient`). Adam hides
  gradient magnitudes (its first step is ~sign), so do not attack Adam updates.
- **Threat model: honest-but-curious server** that reads each client's individual update before
  averaging. Capture it by subclassing Flower's `FedAvg` and overriding `aggregate_train(server_round,
  replies)` (the example `federated/custom_strategy.py` shows how a strategy is customised; it is unused
  reference code that imports `wandb`). `federated/server_app.py` currently uses plain `FedAvg`.
- **Gradient inversion** (`attacks/`): for logistic regression with batch size 1, `grad_w = (p - y) * x`
  and `grad_b = p - y`, so `x = grad_w / grad_b` exactly. With batch B > 1 it is an average (not unique),
  so use gradient matching (DLG-style) and report how success drops with B. Metric: cosine similarity
  between reconstructed and true feature vector (in the client's locally standardized space; mapping back
  to log2(TPM+1) needs the node's mean/std, state that as attacker knowledge). The MLP has no closed form.
- **Membership inference**: counted per patient (4 patients have two samples, all in OTHER's train set).
  Members = a node's train patients; non-members = that node's val/test patients. Metric: attack AUC
  (0.5 = guessing).
- **Defenses**: DP via Opacus (epsilon sweep 1, 3, 8, inf; clip + noise on client gradients) and secure
  aggregation (simulated: the server sees only the sum). Outputs: privacy-utility curve (test AUC vs
  epsilon) and leakage-reduction curve (attack success vs epsilon); look for a knee.
- Open questions: feature set size (try 50 / 100 / 500 genes); whether four nodes (smallest has 34
  train samples) are enough for stable training; whether to report attack success at several batch sizes;
  week 7 proposal framing.

## Decisions and gotchas

- Keep the model simple (logistic regression / small MLP) so the attacks stay interpretable.
- Immunoglobulin genes are excluded on purpose: they are a near-proxy for the IGHV label and
  clone fingerprints. Sex-linked genes are excluded on purpose. X-linked genes are kept.
- Do not compute scaling statistics from validation or test data. Centralized uses pooled train
  statistics; each FL node uses its own train statistics (a known limitation, listed in DATA_NOTES).
- Splits are by patient, never by sample: 4 patients have two samples (all in OTHER's train set).
  Count membership per patient in the membership-inference attack.
- OTHER (UCSD + MDACC + NHLBI) is tiny: 34 train / 8 val / 10 test samples, so its per-node
  metrics are noisy. GCLLSG (34% mutated) differs most from the pooled 51%.
- Seeds should vary model init and batch order; the data split uses `SEED = 4171` in
  `utils/preprocess.py`. Variation from different splits is not captured unless added.
- Gene variance selection uses all 700 samples (label-free); noted as a limitation.
- Epochs (centralized) and rounds (FedAvg) are not the same unit: a round averages four separately
  trained local models. `best_epoch` / `best_round` both mean "chosen by early stopping on validation
  AUC", never from test numbers. The validation AUC itself differs: pooled for centralized, the
  node-size-weighted mean of per-node AUCs for FedAvg.
- FedAvg logistic regression: client-side validation AUC is flat after ~15 rounds, so early stopping
  picks round ~19 (test AUC ~0.955) while the test curve keeps rising to 0.965 at round 200. Part of
  its reported gap to centralized is this selection effect. The user reviewed this and chose to keep
  the protocol unchanged; mention it when comparing numbers.
- Never run `flwr run` here (it builds a separate environment per run and cannot import `utils/` or
  `models/`); the simulation uses `flwr.simulation.run_simulation` in-process. An earlier attempt
  left a Flower SuperLink-default entry in the user's global `~/.flwr/config.toml`.

## Working agreements (how the user likes to work)

- Commit only when asked ("commit"); the user does the pushing.
- For a large new step, first list what will be done and the open choices, then build after the user agrees.
- When the user says "discuss" or "do not make changes", change nothing.
- Keep README.md, CLAUDE.md and DATA_NOTES.txt in sync with the code in the same change.
- Verify claims against the data or the code before stating them, and say plainly when an earlier
  statement was wrong. Explain results in plain language with small tables.
