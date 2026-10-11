"""Shared validation and backend lifecycle for train stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from unirl.distributed.group.remote import Remote
from unirl.train.stack.planner import _positive_int

if TYPE_CHECKING:
    from unirl.algorithms.base import StageAlgorithm
    from unirl.train.backend.fsdp import FSDPBackend


def validate_stack_config(
    *, owner: str, micro_batch_size: int, max_grad_norm: float, num_updates_per_batch: int
) -> tuple[int, float, int]:
    """Validate and normalize the shared scalar constructor arguments."""
    micro_size = int(micro_batch_size)
    if micro_size < 1:
        raise ValueError(f"{owner}.micro_batch_size must be >= 1; got {micro_batch_size}.")
    grad_norm = float(max_grad_norm)
    if grad_norm <= 0.0:
        raise ValueError(f"{owner}.max_grad_norm must be > 0; got {max_grad_norm}.")
    num_updates = _positive_int(name=f"{owner}.num_updates_per_batch", value=num_updates_per_batch)
    return micro_size, grad_norm, num_updates


def validate_multi_update_support(num_updates: int, algorithms: tuple[tuple[str | None, StageAlgorithm], ...]) -> None:
    """Check the algorithms at each stack's existing anchor-validation boundary."""
    if num_updates <= 1:
        return
    for name, algorithm in algorithms:
        if not algorithm.supports_multi_update:
            if name is None:
                raise ValueError(
                    f"num_updates_per_batch={num_updates} requires an algorithm whose "
                    f"old_logp anchor stays frozen across the N optimizer steps "
                    f"(FlowGRPO / FlowDPPO / GRPO / DRPO). "
                    f"{type(algorithm).__name__} sets supports_multi_update=False, so >1 optimizer "
                    f"step would train against a moving anchor. Set num_updates_per_batch=1."
                )
            raise ValueError(
                f"num_updates_per_batch={num_updates} requires every algorithm's "
                f"π_old anchor to stay frozen across the N optimizer steps, but the {name!r} "
                f"algorithm ({type(algorithm).__name__}) sets supports_multi_update=False. Set "
                f"num_updates_per_batch=1."
            )


class TrainStackSupport(Remote):
    """Backend lifecycle shared by stacks with independent training loops."""

    fsdp_backend: FSDPBackend

    def on_rollout_end(self) -> None:
        """Delegate the rollout boundary to the backend's EMA."""
        self.fsdp_backend.on_rollout_end()

    def _train_step_profiler(self):
        """Lazily cache the per-worker train-step profiler, including a disabled profiler."""
        cached = getattr(self, "_profiler_cache", "unset")
        if cached == "unset":
            from unirl.utils.profiling import maybe_build_train_profiler

            cached = maybe_build_train_profiler(int(getattr(self.fsdp_backend, "_rank", 0)))
            self._profiler_cache = cached
        return cached

    def _current_lr(self) -> float:
        optimizer = self.fsdp_backend.optimizer
        param_groups = getattr(optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            return float(param_groups[0]["lr"])
        scheduler = self.fsdp_backend.scheduler
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            last = scheduler.get_last_lr()
            if isinstance(last, list) and last:
                return float(last[0])
        return 0.0
