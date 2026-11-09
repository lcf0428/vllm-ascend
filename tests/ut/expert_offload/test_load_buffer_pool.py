"""Dependency-free threaded tests; run directly with unittest."""

import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/expert_offload/load_buffer_pool.py"
spec = importlib.util.spec_from_file_location("load_buffer_pool", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LoadBufferPool = module.LoadBufferPool


class LoadBufferPoolTests(unittest.TestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(self.executor.shutdown, wait=True)
        self.allocate = Mock(side_effect=bytearray)
        self.pool = LoadBufferPool(256, self.executor, self.allocate)
        self.addCleanup(self.pool.close)

    def submit(self, data, consume):
        size = len(data)
        return self.pool.submit(
            size, lambda storage, offset: memoryview(storage)[offset:offset + size],
            lambda view: view.__setitem__(slice(None), data), consume,
        )

    def test_owns_source_and_workers_execute_concurrently(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        started = [threading.Event(), threading.Event()]
        outputs = [None, None]

        def consume(index, view):
            started[index].set()
            if not gate.wait(5):
                raise TimeoutError("worker gate")
            outputs[index] = bytes(view)

        source = bytearray(b"a" * 100)
        for index in range(2):
            self.submit(source, lambda view, i=index: consume(i, view))
        self.assertTrue(all(event.wait(5) for event in started))
        source[:] = b"b" * 100  # Original checkpoint storage can now change.
        gate.set()
        self.pool.close()
        self.assertEqual(outputs, [b"a" * 100, b"a" * 100])
        self.assertEqual(self.pool.peak_in_use_bytes, 256)
        self.allocate.assert_called_once_with(256)
        self.assertIsNone(self.pool.storage)

    def test_capacity_backpressure_waits_without_another_allocation(self):
        gate, entered, finished = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        errors, outputs = [], []

        def consume(view):
            if not gate.wait(5):
                raise TimeoutError("worker gate")
            outputs.append(bytes(view))

        self.submit(b"a" * 256, consume)

        def producer():
            entered.set()
            try:
                self.submit(b"b" * 128, lambda view: outputs.append(bytes(view)))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertFalse(finished.wait(0.05))
        finally:
            gate.set()
            producer_thread.join(5)
        self.assertFalse(producer_thread.is_alive())
        self.pool.drain()
        self.assertFalse(errors)
        self.assertEqual(outputs, [b"a" * 256, b"b" * 128])
        self.allocate.assert_called_once_with(256)
        self.assertLessEqual(self.pool.peak_in_use_bytes, 256)

    def test_variable_shapes_reuse_one_arena_and_coalesce_ranges(self):
        outputs = []
        expected = []
        for size in [1, 64, 65, 128, 200, 256] * 8:
            data = bytes([size % 255]) * size
            index = len(expected)
            expected.append(data)
            self.submit(data, lambda view, i=index: outputs.append((i, bytes(view))))
        self.pool.drain()
        self.assertEqual([value for _, value in sorted(outputs)], expected)
        self.assertEqual(self.pool.free, [(0, 256)])
        self.assertEqual(self.pool.in_use_bytes, 0)
        self.allocate.assert_called_once_with(256)

    def test_oversized_source_uses_no_staging_and_waits_for_prior_writes(self):
        completed = []
        self.submit(b"a", lambda view: completed.append(bytes(view)))
        stage, consume, view = Mock(), Mock(), Mock()
        self.assertFalse(self.pool.submit(257, view, stage, consume))
        self.assertEqual(completed, [b"a"])
        stage.assert_not_called()
        consume.assert_not_called()
        view.assert_not_called()
        self.assertEqual(self.pool.synchronous, 1)

    def test_oversized_only_workload_does_not_allocate_arena(self):
        self.assertFalse(self.submit(b"a" * 300, Mock()))
        self.allocate.assert_not_called()

    def test_close_waits_for_all_writers_and_propagates_error(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        completed = threading.Event()

        def fail(view):
            if not gate.wait(5):
                raise TimeoutError("worker gate")
            raise RuntimeError("copy failed")

        self.submit(b"a", fail)
        self.submit(b"b", lambda view: completed.set())
        gate.set()
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            self.pool.close()
        self.assertTrue(completed.is_set())
        self.assertFalse(self.pool.pending)
        self.assertIsNone(self.pool.storage)
        self.assertTrue(self.pool.closed)

    def test_stage_failure_releases_lease(self):
        with self.assertRaisesRegex(RuntimeError, "source failed"):
            self.pool.submit(64, lambda storage, offset: memoryview(storage)[:64],
                             Mock(side_effect=RuntimeError("source failed")), Mock())
        self.assertEqual(self.pool.free, [(0, 256)])
        self.assertEqual(self.pool.in_use_bytes, 0)
        self.assertFalse(self.pool.pending)

    def test_task_limit_bounds_metadata_even_for_tiny_shards(self):
        self.pool.max_pending = 1
        outputs = []
        for _ in range(20):
            self.submit(b"a", lambda view: outputs.append(bytes(view)))
            self.assertLessEqual(len(self.pool.pending), 1)
        self.pool.drain()
        self.assertEqual(outputs, [b"a"] * 20)

    def test_closed_pool_rejects_submissions(self):
        self.pool.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.submit(b"a", Mock())


if __name__ == "__main__":
    unittest.main()
