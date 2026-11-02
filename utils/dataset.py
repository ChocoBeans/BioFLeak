"""Read the files written by `python -m utils.preprocess` into arrays."""
import numpy as np
import pandas as pd

from utils.preprocess import PROCESSED_DIR

_ID_COLUMNS = 5  # SAMPLE_ID, PATIENT_ID, COHORT, NODE, label; the gene columns follow


def _read(path):
    df = pd.read_csv(path)
    return (df.iloc[:, _ID_COLUMNS:].to_numpy(dtype=np.float32, copy=True),
            df["label"].to_numpy(dtype=np.int64, copy=True), df["NODE"].to_numpy())


def load_processed():
    """Return (test, nodes) in the same layout as `make_splits`, with unscaled features.

    test = (X, y, node_per_sample); nodes = {node: {"train": (X, y), "val": (X, y)}}.
    Apply `standardize_pooled` (centralized) or `standardize_local` (federated) afterwards.
    """
    nodes = {}
    for folder in sorted((PROCESSED_DIR / "nodes").iterdir()):
        nodes[folder.name] = {s: _read(folder / f"{s}.csv")[:2] for s in ("train", "val")}
    return _read(PROCESSED_DIR / "test.csv"), nodes
