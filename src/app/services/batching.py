"""Generic async dynamic batcher.

Concurrent GPU inference under load is usually bottlenecked by submitting
one sample per forward pass instead of a few batched ones - small-batch GPU
calls are dominated by kernel-launch/Python overhead rather than actual
compute. This groups items submitted concurrently via `submit()` into
batches of up to `max_batch_size`, or whatever has arrived within
`max_wait_s`, and hands each batch to `process_batch` as one call.

Only one batch is ever in flight at a time (`_run` processes them
serially), which is also what bounds GPU concurrency here - no separate
lock/semaphore around the model is needed as long as `process_batch` is the
only thing touching it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from starlette.concurrency import run_in_threadpool

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass
class _Pending(Generic[ItemT, ResultT]):
    item: ItemT
    future: "asyncio.Future[ResultT]"


class DynamicBatcher(Generic[ItemT, ResultT]):
    def __init__(
        self,
        process_batch: Callable[[list[ItemT]], list[ResultT]],
        *,
        max_batch_size: int,
        max_wait_s: float,
    ):
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        self._process_batch = process_batch
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_s
        self._queue: "asyncio.Queue[_Pending[ItemT, ResultT]]" = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        """Start the background batch-consumer loop. Must be called from a
        running event loop; safe to call more than once."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def submit(self, item: ItemT) -> ResultT:
        """Enqueue `item` and wait for its result. Raises whatever exception
        `process_batch` raised for the batch this item ended up in."""
        future: "asyncio.Future[ResultT]" = asyncio.get_running_loop().create_future()
        await self._queue.put(_Pending(item, future))
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            pending = [await self._queue.get()]
            deadline = loop.time() + self._max_wait_s
            while len(pending) < self._max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break

            await self._process(pending)

    async def _process(self, pending: list[_Pending[ItemT, ResultT]]) -> None:
        try:
            results = await run_in_threadpool(self._process_batch, [p.item for p in pending])
        except Exception as exc:
            for p in pending:
                if not p.future.done():
                    p.future.set_exception(exc)
            return

        if len(results) != len(pending):
            error = RuntimeError(
                f"process_batch returned {len(results)} results for a batch of {len(pending)}"
            )
            for p in pending:
                if not p.future.done():
                    p.future.set_exception(error)
            return

        for p, result in zip(pending, results):
            if not p.future.done():
                p.future.set_result(result)
