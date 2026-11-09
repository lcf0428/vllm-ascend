"""Byte-bounded staging for a single checkpoint-loader producer.

Workers only read their leased views and write final destinations. All arena
bookkeeping happens on the producer, which reclaims a lease after its future
completes. A single backing allocation bounds staging storage independently
of tensor shape, dtype, worker count, and checkpoint mapping lifetime.
"""

import time
from concurrent.futures import FIRST_COMPLETED, wait


class LoadBufferPool:
    ALIGNMENT = 64

    def __init__(self, capacity_bytes, executor, allocate, max_pending=128):
        if capacity_bytes <= 0 or max_pending <= 0:
            raise ValueError("Load pool capacity and task limit must be positive")
        self.capacity_bytes = capacity_bytes
        self.executor = executor
        self.allocate = allocate
        self.max_pending = max_pending
        self.storage = None
        self.free = [(0, capacity_bytes)]
        self.pending = {}
        self.closed = False
        self.peak_in_use_bytes = 0
        self.in_use_bytes = 0
        self.submitted = 0
        self.synchronous = 0
        self.wait_seconds = 0.0

    def _release(self, offset, size):
        self.in_use_bytes -= size
        self.free.append((offset, size))
        merged = []
        for start, length in sorted(self.free):
            if merged and merged[-1][0] + merged[-1][1] == start:
                previous, previous_length = merged[-1]
                merged[-1] = (previous, previous_length + length)
            else:
                merged.append((start, length))
        self.free = merged

    def _reap(self, block=False):
        done = [future for future in self.pending if future.done()]
        if not done and block and self.pending:
            started = time.perf_counter()
            done, _ = wait(self.pending, return_when=FIRST_COMPLETED)
            self.wait_seconds += time.perf_counter() - started
        for future in done:
            offset, size = self.pending.pop(future)
            try:
                future.result()
            finally:
                self._release(offset, size)

    def submit(self, nbytes, make_view, stage, consume):
        """Stage synchronously, then consume asynchronously; False means sync fallback.

        make_view(storage, offset) must return an alias of exactly nbytes.
        stage(view) must finish copying the checkpoint source before returning.
        consume(view) must finish writing the final Host destination before
        returning, and must not retain the view. No source tensor is passed to
        the executor. The caller performs oversized copies synchronously.
        """
        if self.closed:
            raise RuntimeError("Load buffer pool is closed")
        if nbytes < 0:
            raise ValueError("Negative load size")
        size = (nbytes + self.ALIGNMENT - 1) // self.ALIGNMENT * self.ALIGNMENT
        self._reap()
        if not size or size > self.capacity_bytes:
            # Also serializes fallback against earlier gate/up shard writes.
            self.drain()
            self.synchronous += 1
            return False
        while True:
            if len(self.pending) < self.max_pending:
                available = next(
                    ((i, start, length) for i, (start, length) in enumerate(self.free) if length >= size), None
                )
                if available is not None:
                    break
            self._reap(block=True)
        if self.storage is None:
            self.storage = self.allocate(self.capacity_bytes)
        index, offset, length = available
        self.free[index:index + 1] = [(offset + size, length - size)] if length > size else []
        self.in_use_bytes += size
        self.peak_in_use_bytes = max(self.peak_in_use_bytes, self.in_use_bytes)
        try:
            view = make_view(self.storage, offset)
            stage(view)
            future = self.executor.submit(consume, view)
        except BaseException:
            self._release(offset, size)
            raise
        self.pending[future] = (offset, size)
        self.submitted += 1
        return True

    def drain(self):
        """Wait for every writer, including on error, before returning/raising."""
        error = None
        for future in list(self.pending):
            offset, size = self.pending.pop(future)
            try:
                future.result()
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                self._release(offset, size)
        if error is not None:
            raise error

    def close(self):
        try:
            self.drain()
        finally:
            self.storage = None
            self.free.clear()
            self.closed = True
