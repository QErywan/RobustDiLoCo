"""
Worker and Simulation classes for single-process DiLoCo simulation.

Each Worker is a Python object with its own model copy, inner optimizer, and
dataloader. Communication is plain Python — no dist.*, no gloo, no torchrun.
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class SimConfig:
    """All hyperparameters for the simulation."""
    # Inner loop
    H: int = 500                          # inner steps per outer step
    inner_lr: float = 6e-4
    inner_weight_decay: float = 0.1
    inner_betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # Outer loop (Nesterov SGD — DiLoCo default)
    outer_lr: float = 0.7
    outer_momentum: float = 0.9

    # Hardware
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    # Move each worker on/off the device one at a time to reduce peak VRAM.
    # Required on GPUs with <16GB when running 8 workers at 124M params.
    offload_between_steps: bool = False
    verbose: bool = False


class Worker:
    """
    A single DiLoCo worker.

    Holds its own model copy, inner AdamW optimizer, outer Nesterov SGD
    optimizer, and dataloader. Runs H inner steps independently, then
    exposes compute_pseudo_grad() and apply_outer_update() for the
    Simulation orchestrator to call.
    """

    def __init__(
        self,
        rank: int,
        model: nn.Module,
        dataloader: DataLoader,
        config: SimConfig,
    ):
        self.rank = rank
        self.config = config
        self.device = torch.device(config.device)

        # Each worker gets its own deep copy of the model.
        # With offload_between_steps, models start on CPU and are paged to the
        # device only during their active step, keeping peak VRAM to one worker.
        init_device = "cpu" if config.offload_between_steps else self.device
        self.model = copy.deepcopy(model).to(init_device)

        self.dataloader = dataloader
        self._data_iter = iter(dataloader)

        self.inner_optimizer = AdamW(
            self.model.parameters(),
            lr=config.inner_lr,
            weight_decay=config.inner_weight_decay,
            betas=config.inner_betas,
        )

        # Outer optimizer — Nesterov SGD, matches DiLoCo paper defaults
        self.outer_optimizer = SGD(
            self.model.parameters(),
            lr=config.outer_lr,
            momentum=config.outer_momentum,
            nesterov=True,
        )

        self._snapshot: Optional[list[torch.Tensor]] = None
        self._param_shapes: list[torch.Size] = [p.shape for p in self.model.parameters()]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _next_batch(self) -> torch.Tensor:
        """Pull the next batch, restarting the iterator when exhausted."""
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            return next(self._data_iter)

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def inner_step(self, steps: Optional[int] = None) -> dict:
        """
        Run H inner AdamW steps on local data. Saves a parameter snapshot
        before the first step so compute_pseudo_grad() can diff against it.

        Returns per-step loss history for logging.
        """
        if steps is None:
            steps = self.config.H

        # Snapshot before any local updates
        self._snapshot = [p.data.detach().clone() for p in self.model.parameters()]

        self.model.train()
        losses = []

        for _ in range(steps):
            batch = self._next_batch()
            input_ids = batch.to(self.device)
            labels = input_ids.clone()

            self.inner_optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.inner_optimizer.step()

            losses.append(loss.item())

        return {"losses": losses, "mean_loss": sum(losses) / len(losses)}

    # ------------------------------------------------------------------
    # Pseudo-gradient
    # ------------------------------------------------------------------

    def compute_pseudo_grad(self) -> torch.Tensor:
        """
        Compute the pseudo-gradient: snapshot − current_params.

        This is the net weight delta from all H inner steps. Returned as a
        single flat tensor so aggregators can treat it as a plain vector.
        """
        if self._snapshot is None:
            raise RuntimeError("call inner_step() before compute_pseudo_grad()")

        deltas = [
            s - p.data
            for s, p in zip(self._snapshot, self.model.parameters())
        ]
        return torch.cat([d.flatten() for d in deltas])

    # ------------------------------------------------------------------
    # Outer update
    # ------------------------------------------------------------------

    def apply_outer_update(self, aggregated_grad: torch.Tensor) -> None:
        """
        Restore params to pre-inner-step snapshot, then apply one Nesterov
        SGD step using the aggregated pseudo-gradient from all workers.

        All workers receive the same aggregated_grad, so they stay in sync.
        """
        if self._snapshot is None:
            raise RuntimeError("call inner_step() before apply_outer_update()")

        # Restore to snapshot — outer step is applied from the same base point
        for p, s in zip(self.model.parameters(), self._snapshot):
            p.data.copy_(s)

        # Unflatten aggregated gradient back to per-parameter shapes and set .grad
        offset = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.grad = (
                aggregated_grad[offset : offset + numel]
                .view(p.shape)
                .to(device=p.device, dtype=p.dtype)
                .clone()
            )
            offset += numel

        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def offload_to(self, device) -> None:
        """
        Move model + both optimizer state dicts to `device`.

        w.model.to(device) alone is insufficient: AdamW/SGD state tensors
        (m, v, momentum buffers) are created on whichever device the parameters
        are on at the time of the first .step() call, and are NOT moved by
        model.to().  Without this helper, all 8 workers' AdamW states accumulate
        on GPU after the first outer step (~8 GB), defeating offload entirely.
        """
        self.model.to(device)
        for optim in (self.inner_optimizer, self.outer_optimizer):
            for state in optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        if self._snapshot is not None:
            self._snapshot = [s.to(device) for s in self._snapshot]

    def grad_norm(self) -> float:
        """L2 norm of current parameter gradients (after outer update)."""
        norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
        return (sum(n ** 2 for n in norms) ** 0.5) if norms else 0.0

    def param_norm(self) -> float:
        """L2 norm of current model parameters."""
        return sum(p.data.norm().item() ** 2 for p in self.model.parameters()) ** 0.5


# ----------------------------------------------------------------------
# Simulation orchestrator
# ----------------------------------------------------------------------

class Simulation:
    """
    Orchestrates a single-process DiLoCo simulation.

    Runs the 5-step outer loop pattern:
        1. Each worker runs H inner steps independently
        2. Collect pseudo-gradients
        3. Apply perturbation (before aggregation)
        4. Aggregate
        5. Broadcast aggregated gradient back to all workers

    The aggregator and perturbation are injected at construction time and
    never hardcoded. Swap them to run different experiments.
    """

    def __init__(self, workers: list[Worker], aggregator, perturbation, config: SimConfig):
        self.workers = workers
        self.aggregator = aggregator
        self.perturbation = perturbation
        self.config = config
        self.outer_step_count = 0

    def run_outer_step(self) -> dict:
        """Run one full outer step and return a metrics dict."""

        # 1. Each worker runs H inner steps independently (no communication)
        worker_metrics = []
        pseudo_grads = []
        for w in self.workers:
            if self.config.verbose:
                print(f"  outer {self.outer_step_count + 1} | worker {w.rank}/{len(self.workers) - 1}", flush=True)
            if self.config.offload_between_steps:
                w.offload_to(w.device)
            m = w.inner_step(steps=self.config.H)
            worker_metrics.append(m)
            pseudo_grads.append(w.compute_pseudo_grad())
            if self.config.offload_between_steps:
                w.offload_to("cpu")

        # 2. Collect pseudo-gradients — one flat tensor per worker (already done above if offloading)

        # Record norms before perturbation for diagnostics
        pg_norms_before = [g.norm().item() for g in pseudo_grads]

        # 3. Perturbation injection — BEFORE aggregation
        pseudo_grads = self.perturbation.apply(pseudo_grads)

        # 4. Aggregation
        aggregated = self.aggregator.aggregate(pseudo_grads)

        # 5. Apply outer update to all workers
        for w in self.workers:
            if self.config.offload_between_steps:
                w.offload_to(w.device)
            w.apply_outer_update(aggregated)
            if self.config.offload_between_steps:
                w.offload_to("cpu")

        self.outer_step_count += 1

        # Build metrics dict for the training loop to log
        mean_losses = [m["mean_loss"] for m in worker_metrics]
        return {
            "outer_step": self.outer_step_count,
            "loss/mean": sum(mean_losses) / len(mean_losses),
            "loss/min": min(mean_losses),
            "loss/max": max(mean_losses),
            "pseudo_grad_norm/mean": sum(pg_norms_before) / len(pg_norms_before),
            "pseudo_grad_norm/max": max(pg_norms_before),
            "aggregated_grad_norm": aggregated.norm().item(),
            "worker_losses": mean_losses,
        }

    @property
    def global_model(self) -> nn.Module:
        """Return worker 0's model as the reference global model (all are in sync)."""
        return self.workers[0].model
