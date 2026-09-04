"""Generic batch + concurrency + retry runner for LLM calls.

Every task that calls an LLM in batches (dataset generation, text
relabeling, multimodal relabeling) goes through `run_batches` instead of
each hand-rolling its own ThreadPoolExecutor/retry loop - this is the single
place batch size, concurrency, and retry/backoff behavior are adjusted.
"""

from __future__ import annotations

import concurrent.futures
import random
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class BatchJobConfig:
    batch_size: int
    max_concurrency: int = 1
    max_retries: int = 2
    retry_delay_s: float = 2.0


@dataclass(frozen=True)
class BatchOutcome(Generic[T, R]):
    batch_number: int
    batch: list[T]
    result: R


def chunk(items: list[T], batch_size: int) -> list[list[T]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def with_retries(
    fn: Callable[[], R],
    *,
    max_retries: int,
    retry_delay_s: float,
    label: str = "",
) -> R:
    """Call `fn`, retrying on any exception with exponential backoff + jitter.
    Re-raises the last exception once retries are exhausted."""
    last_error: Exception | None = None
    total_attempts = max_retries + 1
    prefix = f"[data_prep]{' ' + label if label else ''}"
    for attempt in range(1, total_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < total_attempts:
                delay = retry_delay_s * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(
                    f"{prefix} attempt {attempt}/{total_attempts} failed "
                    f"({type(e).__name__}: {e}), retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def run_batches(
    items: list[T],
    *,
    job_config: BatchJobConfig,
    process_batch: Callable[[int, list[T]], R],
    on_batch_done: Callable[[BatchOutcome[T, R]], None] | None = None,
) -> list[BatchOutcome[T, R]]:
    """Split `items` into batches of `job_config.batch_size` and run
    `process_batch(batch_number, batch)` over them.

    - `job_config.max_concurrency <= 1` runs batches sequentially in order.
    - `job_config.max_concurrency > 1` runs them concurrently via a thread
      pool (appropriate here since each batch is one blocking network call).
    - Each batch is retried up to `job_config.max_retries` times with
      exponential backoff; `process_batch` should simply raise on failure -
      retrying is handled here, not by the caller.
    - `on_batch_done`, if given, is called as soon as each batch finishes
      (in completion order, not necessarily batch order) - use it to
      checkpoint results durably before the whole run completes.

    Returns outcomes in original batch order.
    """
    batches = chunk(items, job_config.batch_size)
    outcomes: list[BatchOutcome[T, R] | None] = [None] * len(batches)

    def _run_one(batch_number: int, batch: list[T]) -> R:
        return with_retries(
            lambda: process_batch(batch_number, batch),
            max_retries=job_config.max_retries,
            retry_delay_s=job_config.retry_delay_s,
            label=f"batch {batch_number + 1}/{len(batches)}",
        )

    if job_config.max_concurrency <= 1:
        for i, batch in enumerate(batches):
            outcome = BatchOutcome(i, batch, _run_one(i, batch))
            outcomes[i] = outcome
            if on_batch_done:
                on_batch_done(outcome)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=job_config.max_concurrency) as executor:
            futures = {executor.submit(_run_one, i, batch): (i, batch) for i, batch in enumerate(batches)}
            for future in concurrent.futures.as_completed(futures):
                i, batch = futures[future]
                outcome = BatchOutcome(i, batch, future.result())
                outcomes[i] = outcome
                if on_batch_done:
                    on_batch_done(outcome)

    return outcomes  # type: ignore[return-value]
