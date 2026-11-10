"""Client-side differential privacy: DP-SGD local steps with Opacus.

Each client clips every per-sample gradient to norm `clip`, adds Gaussian noise (std noise_multiplier *
clip) to the summed gradient and takes a plain SGD step. The noise multiplier is calibrated once per
client so that its whole training run (rounds x local steps, sampling rate batch/n) costs (epsilon,
delta) under Opacus' RDP accountant, i.e. record-level DP per site. Batches are drawn without
replacement at a fixed size (as in the non-private runs) while the accountant assumes Poisson sampling,
the usual practical approximation. Epsilon <= 0 means no DP.
"""
import math

import torch
from torch import nn

DELTA, CLIP = 1e-5, 1.0


def local_plan(n, batch_size, local_steps):
    """(batch size, steps per round, sampling rate) of a client with n training samples."""
    batch = min(batch_size, n)
    steps = local_steps or math.ceil(n / batch)  # local_steps = 0: one pass over the data
    return batch, steps, batch / n


def noise_multiplier(epsilon, rounds, n, batch_size, local_steps, delta=DELTA):
    from opacus.accountants.utils import get_noise_multiplier
    _, steps, rate = local_plan(n, batch_size, local_steps)
    return get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, sample_rate=rate,
                                steps=rounds * steps, accountant="rdp")


def train_steps_dp(model, X, y, lr, steps, batch_size, sigma, clip=CLIP, generator=None):
    """`steps` DP-SGD steps on random batches; returns the mean (unclipped) training loss."""
    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer
    from utils.train import sample_batches
    model.train()
    wrapped = GradSampleModule(model)
    optimizer = DPOptimizer(torch.optim.SGD(model.parameters(), lr=lr), noise_multiplier=sigma,
                            max_grad_norm=clip, expected_batch_size=batch_size, generator=generator)
    X, y = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    for idx in sample_batches(len(X), steps, batch_size, generator):
        optimizer.zero_grad()
        loss = loss_fn(wrapped(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / steps
