"""
Instrumented DiLoCo simulation for pseudo-gradient geometry analysis.

InstrumentedSimulation is a subclass of Simulation that fully overrides
run_outer_step to capture three measurement points per step:
  1. Before perturbation: per-worker norms, honest cluster cosine sim,
     Byzantine cosine sim relative to oracle, oracle honest mean.
  2. After perturbation: per-worker norms (shows attack effect).
  3. After aggregation: cosine similarity between aggregate and oracle.

At designated PCA steps, projects all 8 pseudo-grad vectors into a 2D
PCA space computed in-memory (no full tensors written to disk).

Production code (workers.py, run_experiment.py, sweep.py) is not modified.
"""

import json
from pathlib import Path

import torch
from torch import Tensor

from simulation.workers import Simulation, SimConfig, Worker


class InstrumentedSimulation(Simulation):
    """
    Subclass of Simulation that captures geometric metrics of pseudo-gradients
    at three measurement points per outer step.

    Production code is untouched — this overrides run_outer_step only.
    Byzantine workers are the last `byzantine_f` workers in the list,
    consistent with simulation/perturbations.py.
    """

    def __init__(
        self,
        workers: list[Worker],
        aggregator,
        perturbation,
        config: SimConfig,
        out_dir: Path,
        byzantine_f: int = 0,
        pca_steps: set[int] | None = None,
    ):
        """
        Args:
            workers: list of Worker objects (same as Simulation).
            aggregator: aggregation rule (same as Simulation).
            perturbation: perturbation to inject (same as Simulation).
            config: SimConfig (same as Simulation).
            out_dir: directory for output JSON files.
            byzantine_f: number of Byzantine workers (last f in the list).
            pca_steps: outer steps at which to compute and store PCA snapshots.
                       Defaults to {1, 10, 25, 50}.
        """
        super().__init__(workers, aggregator, perturbation, config)
        self.out_dir = Path(out_dir)
        self.byzantine_f = byzantine_f
        self.pca_steps = pca_steps if pca_steps is not None else {1, 10, 25, 50}
        self.analysis_history: list[dict] = []
        self._pca_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def run_outer_step(self) -> dict:
        """
        Full override of Simulation.run_outer_step. Adds three measurement
        points and PCA snapshots at designated steps. The return dict is a
        superset of the parent's return dict — all downstream logging code
        that reads parent keys continues to work unchanged.
        """
        # --- Inner steps (replicate parent's offload logic) ---
        worker_metrics = []
        pseudo_grads_before: list[Tensor] = []

        for w in self.workers:
            if self.config.verbose:
                print(
                    f"  outer {self.outer_step_count + 1} | "
                    f"worker {w.rank}/{len(self.workers) - 1}",
                    flush=True,
                )
            if self.config.offload_between_steps:
                w.model.to(w.device)
            m = w.inner_step(steps=self.config.H)
            worker_metrics.append(m)
            # Move to CPU immediately so all 8 pseudo-grad vectors never live
            # on the GPU at once (each is ~500 MB for a 124M-param model).
            pseudo_grads_before.append(w.compute_pseudo_grad().cpu())
            if self.config.offload_between_steps:
                w.model.to("cpu")

        # --- Perturbation ---
        pseudo_grads_after = self.perturbation.apply(pseudo_grads_before)

        # --- Aggregation ---
        aggregated = self.aggregator.aggregate(pseudo_grads_after)

        # --- PCA snapshot at designated steps (before perturbation) ---
        current_step = self.outer_step_count + 1
        if current_step in self.pca_steps:
            # oracle for PCA: honest workers only (pre-perturbation)
            n = len(self.workers)
            n_honest = n - self.byzantine_f
            oracle = torch.stack(pseudo_grads_before[:n_honest]).mean(dim=0)
            self._save_pca(pseudo_grads_before, oracle, aggregated, current_step)

        # --- Analysis metrics ---
        n = len(self.workers)
        n_honest = n - self.byzantine_f
        oracle = torch.stack(pseudo_grads_before[:n_honest]).mean(dim=0)
        step_metrics = self._collect_metrics(
            pseudo_grads_before, pseudo_grads_after, aggregated, current_step
        )

        # --- Outer updates (replicate parent's offload logic) ---
        for w in self.workers:
            if self.config.offload_between_steps:
                w.model.to(w.device)
            w.apply_outer_update(aggregated)
            if self.config.offload_between_steps:
                w.model.to("cpu")

        self.outer_step_count += 1

        # --- Build full metrics dict (superset of parent's keys) ---
        mean_losses = [m["mean_loss"] for m in worker_metrics]
        per_worker_norms_before = step_metrics["per_worker_norms_before"]
        metrics = {
            "outer_step": self.outer_step_count,
            "loss/mean": sum(mean_losses) / len(mean_losses),
            "loss/min": min(mean_losses),
            "loss/max": max(mean_losses),
            "pseudo_grad_norm/mean": sum(per_worker_norms_before) / n,
            "pseudo_grad_norm/max": max(per_worker_norms_before),
            "aggregated_grad_norm": aggregated.norm().item(),
            "worker_losses": mean_losses,
            **step_metrics,
        }
        self.analysis_history.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Metric helpers (factored out so tests can call without a training loop)
    # ------------------------------------------------------------------

    def _collect_metrics(
        self,
        pseudo_grads_before: list[Tensor],
        pseudo_grads_after: list[Tensor],
        aggregated: Tensor,
        step: int,
    ) -> dict:
        """
        Compute analysis metrics from synthetic or real pseudo-grad lists.
        Called inside run_outer_step; also directly callable from tests.
        """
        n = len(pseudo_grads_before)
        n_honest = n - self.byzantine_f

        # Point 1: before perturbation
        per_worker_norms_before = [g.norm().item() for g in pseudo_grads_before]
        oracle = torch.stack(pseudo_grads_before[:n_honest]).mean(dim=0)

        honest_grads = pseudo_grads_before[:n_honest]
        if n_honest >= 2:
            pairs = [
                (i, j)
                for i in range(n_honest)
                for j in range(i + 1, n_honest)
            ]
            honest_cosine_sim = sum(
                self._cosine(honest_grads[i], honest_grads[j]) for i, j in pairs
            ) / len(pairs)
        else:
            honest_cosine_sim = 1.0

        byz_grads = pseudo_grads_before[n_honest:]
        byzantine_cosine_sim = (
            sum(self._cosine(g, oracle) for g in byz_grads) / len(byz_grads)
            if byz_grads
            else 0.0
        )

        # Point 2: after perturbation
        per_worker_norms_after = [g.norm().item() for g in pseudo_grads_after]

        # Point 3: after aggregation
        cosine_to_oracle = self._cosine(aggregated, oracle)

        return {
            "per_worker_norms_before": per_worker_norms_before,
            "per_worker_norms_after": per_worker_norms_after,
            "honest_cosine_sim": honest_cosine_sim,
            "byzantine_cosine_sim": byzantine_cosine_sim,
            "cosine_to_oracle": cosine_to_oracle,
        }

    def _save_pca(
        self,
        pseudo_grads_before: list[Tensor],
        oracle: Tensor,
        aggregated: Tensor,
        step: int,
    ) -> None:
        """
        Compute 2D PCA of the (n, d) pre-perturbation pseudo-grad matrix
        in-memory and buffer the (n, 2) projections in self._pca_cache.
        Full tensors are never written to disk.
        """
        stacked = torch.stack(pseudo_grads_before)   # (n, d)
        center = stacked.mean(dim=0)                  # (d,)
        centered = stacked - center                   # (n, d)

        # pca_lowrank: (U, S, V) where V is (d, 2)
        _, _, V = torch.pca_lowrank(centered, q=2, center=False)

        worker_projections = (centered @ V).tolist()                        # (n, 2)
        oracle_projection = ((oracle - center) @ V).tolist()               # (2,)
        aggregated_projection = ((aggregated - center) @ V).tolist()       # (2,)

        self._pca_cache[step] = {
            "step": step,
            "worker_projections": worker_projections,
            "oracle_projection": oracle_projection,
            "aggregated_projection": aggregated_projection,
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_metrics(self, agg_name: str) -> None:
        """
        Write analysis_history to metrics_{agg_name}.json and all buffered
        PCA snapshots to pca_{agg_name}_step{N}.json in out_dir.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = self.out_dir / f"metrics_{agg_name}.json"
        with open(metrics_path, "w") as f:
            json.dump(self.analysis_history, f, indent=2)

        for step, pca_data in self._pca_cache.items():
            pca_path = self.out_dir / f"pca_{agg_name}_step{step}.json"
            with open(pca_path, "w") as f:
                json.dump(pca_data, f, indent=2)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: Tensor, b: Tensor) -> float:
        """Cosine similarity between two flat tensors. Returns 0.0 if either is zero."""
        denom = a.norm() * b.norm()
        if denom.item() < 1e-12:
            return 0.0
        return (torch.dot(a.flatten(), b.flatten()) / denom).item()
