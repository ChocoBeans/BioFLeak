"""FedAvg that records each client's individual update, as an honest-but-curious server could.

The server sees every client's returned weights before averaging. `CaptureFedAvg` stores, per
round, the global model it sent and each client's update (returned weights minus global weights),
flattened in `state_dict` order, and writes them to an .npz for the attacks. The ground-truth
batches are not stored: the attacks recompute them from the seeded generators.
"""
import numpy as np
import torch
from flwr.serverapp.strategy import FedAvg


def flatten(arrays):
    """ArrayRecord -> one float64 vector, in state_dict order."""
    return torch.cat([v.flatten().double() for v in arrays.to_torch_state_dict().values()]).numpy()


class CaptureFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_by_round, self.delta_by_round = {}, {}

    def configure_train(self, server_round, arrays, config, grid):
        self.global_by_round[server_round] = flatten(arrays)
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        replies = [m for m in replies if m.has_content()]
        deltas = {}
        for msg in replies:
            metrics = msg.content["metrics"]
            partition = int(metrics["partition-id"])
            del metrics["partition-id"]  # bookkeeping only; keep it out of the averaged metrics
            deltas[partition] = flatten(msg.content["arrays"]) - self.global_by_round[server_round]
        self.delta_by_round[server_round] = deltas
        return super().aggregate_train(server_round, replies)

    def save(self, path, **extra):
        """Write global (R, P) and delta (R, C, P); clients are indexed by partition id."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rounds = sorted(self.delta_by_round)
        n_clients = max(max(d) for d in self.delta_by_round.values()) + 1
        delta = np.full((len(rounds), n_clients, len(self.global_by_round[1])), np.nan)
        for i, r in enumerate(rounds):
            for c, d in self.delta_by_round[r].items():
                delta[i, c] = d
        np.savez(path, rounds=np.array(rounds), delta=delta,
                 glob=np.stack([self.global_by_round[r] for r in rounds]), **extra)
