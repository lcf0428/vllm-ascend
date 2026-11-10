"""Dependency-free ownership tests; also runnable directly with unittest.

Manager adapters are extracted from their actual AST to test them without
importing vLLM/torch_npu. This checks orchestration, not accelerator kernels.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from enum import Enum
import importlib.util
from pathlib import Path
import sys
import time
import random
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/expert_offload"
spec = importlib.util.spec_from_file_location("exclusive_pool", SOURCE / "exclusive_pool.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ExclusiveExpertPool = module.ExclusiveExpertPool
load_spec = importlib.util.spec_from_file_location("load_buffer_pool", SOURCE / "load_buffer_pool.py")
load_module = importlib.util.module_from_spec(load_spec)
load_spec.loader.exec_module(load_module)
LoadBufferPool = load_module.LoadBufferPool


def method(name, **namespace):
    namespace.setdefault("ACL_FORMAT_FRACTAL_NZ", 29)
    tree = ast.parse((SOURCE / "expert_offload_manager.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExpertOffloadManager")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    body = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                            function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(body), str(SOURCE), "exec"), namespace)
    return namespace[name]


class Buffer:
    dtype = "uint8"

    def __init__(self, value, size=4):
        self.shape = (size,)
        self.value = bytes([value] * size)

    def untyped_storage(self):
        return self

    def nbytes(self):
        return len(self.value)

    def copy_(self, other, **kwargs):
        self.value = other.value


def bundle(eid, size=4):
    return {name: Buffer(eid + offset, size) for offset, name in enumerate(
        ("w13", "w2", "w13_weight_scale", "w2_weight_scale", "w13_weight_offset", "w2_scale_bias"))}


class ExclusivePoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = ExclusiveExpertPool()
        self.pool.add_layer(0, [bundle(i) for i in range(6)], [0, 1, -1, -1, -1, -1])
        self.device = {0: bundle(0), 1: bundle(1)}

    def exchange(self, slot, eid):
        def copy(destination, source):
            for name in destination:
                destination[name].copy_(source[name])
        return self.pool.swap(0, slot, eid,
                              lambda target: copy(target, self.device[slot]),
                              lambda source: copy(self.device[slot], source))

    def test_compact_capacity_and_repeated_byte_exact_roundtrips(self):
        initial_addresses = {id(t) for b in list(self.pool.host.values()) + list(self.pool.spares.values())
                             for t in b.values()}
        for slot, eid in [(0, 2), (1, 3), (0, 0), (1, 1), (0, 5), (0, 2)] * 5:
            if self.pool.resident[0, slot] == eid:
                continue
            self.exchange(slot, eid)
            self.assertEqual(len(self.pool.host), 4)
            self.assertEqual(len(self.pool.spares), 1)
            self.assertEqual(self.pool.resident[0, slot], eid)
            for name, tensor in self.device[slot].items():
                self.assertEqual(tensor.value, bundle(eid)[name].value)
            for (_, host_eid), host_bundle in self.pool.host.items():
                for name, tensor in host_bundle.items():
                    self.assertEqual(tensor.value, bundle(host_eid)[name].value)
            addresses = {id(t) for b in list(self.pool.host.values()) + list(self.pool.spares.values())
                         for t in b.values()}
            self.assertEqual(addresses, initial_addresses)

    def test_spare_shared_across_layers_by_layout(self):
        self.pool.add_layer(1, [bundle(i) for i in range(6)], [0, 1, -1, -1, -1, -1])
        self.assertEqual(len(self.pool.spares), 1)
        self.pool.add_layer(2, [bundle(i, 8) for i in range(6)], [0, 1, -1, -1, -1, -1])
        self.assertEqual(len(self.pool.spares), 2)

    def test_mapping_committed_only_after_both_transfers(self):
        stages = []
        def transfer(_):
            self.assertEqual(self.pool.resident[0, 0], 0)
            self.assertIn((0, 2), self.pool.host)
            stages.append(1)
        self.pool.swap(0, 0, 2, transfer, transfer)
        self.assertEqual(len(stages), 2)
        self.assertNotIn((0, 2), self.pool.host)
        self.assertIn((0, 0), self.pool.host)

    def test_failure_poisoning_for_both_directions(self):
        for failing_stage in (0, 1):
            pool = ExclusiveExpertPool()
            pool.add_layer(0, [bundle(i) for i in range(3)], [0, -1, -1])
            callbacks = [Mock(), Mock()]
            callbacks[failing_stage].side_effect = RuntimeError("DMA failed")
            with self.assertRaisesRegex(RuntimeError, "DMA failed"):
                pool.swap(0, 0, 1, *callbacks)
            self.assertEqual(pool.resident[0, 0], 0)
            self.assertTrue(pool.failed)
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                pool.source(0, 1)
            if failing_stage == 0:
                callbacks[1].assert_not_called()

    def test_invalid_initial_mapping_rejected(self):
        for mapping in ([0, 0, -1], [1, -1, -1], [-1, -1, -1]):
            with self.assertRaises(ValueError):
                ExclusiveExpertPool().add_layer(0, [bundle(i) for i in range(3)], mapping)

    def test_manager_swap_preserves_all_fields_and_synchronizes(self):
        events = []
        def task(source, destination, nbytes, **kwargs):
            return SimpleNamespace(source=source, destination=destination, nbytes=nbytes, **kwargs)
        def tasks(layer, idx, eid, slot, host_bundle=None):
            return [task(t, self.device[slot][name], t.nbytes(), name=name)
                    for name, t in host_bundle.items()]
        def copy_batch(batch):
            events.append("batch")
            for item in batch:
                item.destination.copy_(item.source)
        manager = SimpleNamespace(
            _exclusive_pool=self.pool, _build_expert_h2d_tasks=tasks, load_stream=object(),
            _get_h2d_transport=lambda: SimpleNamespace(copy_batch=copy_batch),
            _synchronize_h2d=lambda: events.append("sync"),
            _refresh_expert_fp32_scale=lambda *args: events.append("refresh"))
        swap_batch = method("_exclusive_swap_batch", H2DCopyTask=task, CopyDirection=SimpleNamespace(D2H="d2h"),
                            torch_npu=SimpleNamespace(npu=SimpleNamespace(stream=lambda _: nullcontext())))
        manager._exclusive_swap_batch = lambda *args: swap_batch(manager, *args)
        swap = method("_exclusive_swap")
        for eid in (2, 0, 4, 2):
            events.clear()
            swap(manager, object(), 0, 0, eid)
            self.assertEqual(events, ["batch", "refresh", "sync"])
            for name, tensor in self.device[0].items():
                self.assertEqual(tensor.value, bundle(eid)[name].value)

    def test_batch_deferred_stream_preserves_bytes_over_random_swaps(self):
        rng = random.Random(421)
        self.pool = ExclusiveExpertPool()
        self.pool.add_layer(0, [bundle(i) for i in range(12)], list(range(5)) + [-1] * 7)
        self.device = {slot: bundle(slot) for slot in range(5)}
        original_addresses = {id(t) for b in list(self.pool.host.values()) + list(self.pool.spares.values())
                              for t in b.values()}
        for _ in range(200):
            slots = rng.sample(list(range(5)), rng.randint(1, 5))
            incoming = rng.sample([eid for layer, eid in self.pool.host if layer == 0], len(slots))
            loads = list(zip(slots, incoming))
            original_host, original_resident = dict(self.pool.host), dict(self.pool.resident)
            queue, syncs = [], []

            def prepare(plan):
                self.assertEqual(self.pool.host, original_host)
                for previous, current in zip(plan, plan[1:]):
                    self.assertIs(current.destination, previous.source)
                tasks = []
                for step in plan:
                    for name in step.destination:
                        tasks.append((step.destination[name], self.device[step.slot][name]))
                    for name in step.source:
                        tasks.append((self.device[step.slot][name], step.source[name]))
                return tasks

            def synchronize():
                syncs.append(1)
                self.assertEqual(self.pool.host, original_host)
                self.assertEqual(self.pool.resident, original_resident)
                # Read the live source only now, after ALL tasks were queued.
                # Grouping all D2H before H2D would corrupt the second source.
                for destination, source in queue:
                    destination.copy_(source)

            self.pool.swap_batch(0, loads, prepare, queue.extend, synchronize)
            self.assertEqual(syncs, [1])
            for slot, eid in self.pool.resident.items():
                for name, tensor in self.device[slot[1]].items():
                    self.assertEqual(tensor.value, bundle(eid)[name].value)
            for (_, eid), host in self.pool.host.items():
                for name, tensor in host.items():
                    self.assertEqual(tensor.value, bundle(eid)[name].value)
            self.assertEqual(len(self.pool.host), 7)
            self.assertEqual(original_addresses, {id(t) for b in list(self.pool.host.values())
                                                  + list(self.pool.spares.values()) for t in b.values()})

    def test_batch_shared_spare_is_locked_until_completion_across_layers(self):
        self.pool.add_layer(1, [bundle(i) for i in range(6)], [0, 1, -1, -1, -1, -1])
        first_waiting, release, second_started, second_prepared = [threading.Event() for _ in range(4)]
        errors = []
        def sync_first():
            first_waiting.set()
            if not release.wait(5):
                raise TimeoutError("completion gate")
        def run(layer):
            try:
                if layer == 1:
                    second_started.set()
                self.pool.swap_batch(layer, [(0, 2)],
                                    lambda plan: second_prepared.set() if layer == 1 else plan,
                                    lambda prepared: None, sync_first if layer == 0 else lambda: None)
            except BaseException as error:
                errors.append(error)
        first = threading.Thread(target=run, args=(0,))
        second = threading.Thread(target=run, args=(1,))
        first.start()
        try:
            self.assertTrue(first_waiting.wait(5))
            second.start()
            self.assertTrue(second_started.wait(5))
            self.assertFalse(second_prepared.wait(0.05))
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(errors)
        self.assertTrue(second_prepared.is_set())
        self.assertFalse(first.is_alive() or second.is_alive())

    def test_batch_rejects_invalid_plan_before_any_submission(self):
        for loads in ([(0, 2), (0, 3)], [(0, 2), (1, 2)], [(0, 1)], [(9, 2)], [(0, 99)]):
            prepare, enqueue, sync = Mock(), Mock(), Mock()
            with self.subTest(loads=loads), self.assertRaises((ValueError, KeyError)):
                self.pool.swap_batch(0, loads, prepare, enqueue, sync)
            prepare.assert_not_called()
            enqueue.assert_not_called()
            sync.assert_not_called()
            self.assertFalse(self.pool.failed)

    def test_batch_preparation_failure_is_nondestructive(self):
        enqueue, sync = Mock(), Mock()
        with self.assertRaisesRegex(RuntimeError, "descriptor"):
            self.pool.swap_batch(0, [(0, 2), (1, 3)], Mock(side_effect=RuntimeError("descriptor")), enqueue, sync)
        enqueue.assert_not_called()
        sync.assert_not_called()
        self.assertFalse(self.pool.failed)
        self.assertEqual(self.pool.resident, {(0, 0): 0, (0, 1): 1})

    def test_batch_failure_at_every_transfer_boundary_never_commits(self):
        for failing_index in range(24):
            pool = ExclusiveExpertPool()
            pool.add_layer(0, [bundle(i) for i in range(6)], [0, 1, -1, -1, -1, -1])
            device = {0: bundle(0), 1: bundle(1)}
            original_host, original_resident, original_spares = dict(pool.host), dict(pool.resident), dict(pool.spares)
            queue = []

            def prepare(plan):
                return [(dst[name], src[name]) for step in plan
                        for dst, src in ((step.destination, device[step.slot]), (device[step.slot], step.source))
                        for name in dst]

            def enqueue(tasks):
                for index, task in enumerate(tasks):
                    if index == failing_index:
                        raise RuntimeError("submit failed")
                    queue.append(task)

            def sync():
                for dst, src in queue:
                    dst.copy_(src)
                queue.clear()

            with self.subTest(failing_index=failing_index), self.assertRaisesRegex(RuntimeError, "submit failed"):
                pool.swap_batch(0, [(0, 2), (1, 3)], prepare, enqueue, sync)
            self.assertFalse(queue)
            self.assertTrue(pool.failed)
            self.assertEqual(pool.host, original_host)
            self.assertEqual(pool.resident, original_resident)
            self.assertEqual(pool.spares, original_spares)
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                pool.swap_batch(0, [], Mock(), Mock(), Mock())

    def test_batch_completion_failure_poisoning_and_empty_batch(self):
        enqueue, sync = Mock(), Mock(side_effect=RuntimeError("completion failed"))
        self.pool.swap_batch(0, [], Mock(), enqueue, sync)
        enqueue.assert_not_called()
        sync.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "completion failed"):
            self.pool.swap_batch(0, [(0, 2)], lambda plan: plan, enqueue, sync)
        self.assertTrue(self.pool.failed)
        self.assertEqual(self.pool.resident, {(0, 0): 0, (0, 1): 1})

    def test_manager_batches_two_experts_with_one_completion_barrier(self):
        pending, events = [], []
        def task(source, destination, nbytes, **kwargs):
            return SimpleNamespace(source=source, destination=destination, nbytes=nbytes, **kwargs)
        def build(layer, idx, eid, slot, host_bundle=None):
            events.append("build")
            return [task(t, self.device[slot][name], t.nbytes(), name=name) for name, t in host_bundle.items()]
        def enqueue(tasks):
            events.append("enqueue")
            pending.extend(tasks)
        def synchronize():
            events.append("sync")
            self.assertEqual(self.pool.resident, {(0, 0): 0, (0, 1): 1})
            for t in pending:
                t.destination.copy_(t.source)
        manager = SimpleNamespace(
            _exclusive_pool=self.pool, load_stream=object(), _build_expert_h2d_tasks=build,
            _get_h2d_transport=lambda: SimpleNamespace(copy_batch=enqueue), _synchronize_h2d=synchronize,
            _refresh_expert_fp32_scale=lambda *args: events.append("scale"))
        batch = method("_exclusive_swap_batch", H2DCopyTask=task, CopyDirection=SimpleNamespace(D2H="d2h"),
                       torch_npu=SimpleNamespace(npu=SimpleNamespace(stream=lambda _: nullcontext())))
        manager._exclusive_swap_batch = lambda *args: batch(manager, *args)
        method("_load_expert_weights_into_slots")(manager, object(), 0, [(0, 2), (1, 3)])
        self.assertEqual(events, ["build"] * 4 + ["enqueue", "scale", "scale", "sync"])
        for slot, eid in ((0, 2), (1, 3)):
            for name, t in self.device[slot].items():
                self.assertEqual(t.value, bundle(eid)[name].value)

    def test_prefill_reads_residents_from_device_and_misses_from_host(self):
        manager = SimpleNamespace(
            num_total_experts=6, offload_config=SimpleNamespace(prefill_split_gmm=False),
            _build_prefill_local_d2d_tasks=lambda *args: [("device", args[-1])],
            _build_prefill_h2d_tasks=lambda layer, pool, eid, dst: [("host", eid)])
        build = method("_build_prefill_load_tasks")
        for mapping in ([0, 1, -1, -1, -1, -1], [-1, 1, 0, -1, -1, -1]):
            tasks, h2d, d2d, skip = build(manager, 0, 0, mapping)
            self.assertEqual((h2d, d2d, skip), (4, 2, 0))
            self.assertEqual(tasks, [("device" if s >= 0 else "host", eid) for eid, s in enumerate(mapping)])

    def test_exclusive_prefill_disables_background_readers(self):
        manager = SimpleNamespace(_exclusive_pool=self.pool, _prefill_initialized=True, _skip_prefill=False)
        active = method("_prefill_pipe_active")
        self.assertFalse(active(manager))
        manager._exclusive_pool = None
        self.assertTrue(active(manager))

    def test_graph_callback_reexecutes_and_fails_closed(self):
        logger, os = Mock(), Mock()
        callback = method("_update_exclusive_weights_callback", logger=logger, os=os)
        manager = SimpleNamespace(_update_weights=Mock())
        for route in (2, 0, 3):
            callback(manager, route)
        self.assertEqual([call.args[0] for call in manager._update_weights.call_args_list], [2, 0, 3])
        os._exit.assert_not_called()
        manager._update_weights.side_effect = RuntimeError("partial overwrite")
        callback(manager, 4)
        os._exit.assert_called_once_with(1)

    def test_exclusive_configuration_guards(self):
        # Execute the actual defaults/constructor; isolate the existing general
        # validation (which imports optional predictor/runtime dependencies).
        tree = ast.parse((ROOT / "vllm_ascend/ascend_config.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExpertOffloadConfig")
        cls.body = [node for node in cls.body if isinstance(node, ast.Assign)
                    or isinstance(node, ast.FunctionDef) and node.name in ("__init__", "__getattr__")]
        namespace = {"CpuMoeConfig": lambda config: SimpleNamespace(enabled=bool((config or {}).get("enabled")))}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "config", "exec"), namespace)
        config_type = namespace["ExpertOffloadConfig"]
        config_type._validate_config = lambda self: None
        self.assertFalse(config_type({}).exclusive_expert_storage)
        self.assertFalse(config_type({}).exclusive_load_pool_enabled)
        self.assertEqual(config_type({}).exclusive_load_pool_size_mb, 1024)
        valid = {"expert_offload": True, "exclusive_expert_storage": True}
        self.assertTrue(config_type(valid).prefill_delta_load)
        for invalid in ({"cpu_moe": {"enabled": True}}, {"enable_multi_card": True},
                        {"expert_prefetch_enabled": True}, {"h2d_backend": "memfabric"},
                        {"prefill_split_gmm": True}, {"expert_offload": False}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config_type(valid | invalid)
        with self.assertRaises(TypeError):
            config_type(valid | {"exclusive_expert_storage": "true"})
        pool_config = config_type(valid | {"exclusive_load_pool_enabled": True, "exclusive_load_pool_size_mb": 64})
        self.assertEqual(pool_config.exclusive_load_pool_size_mb, 64)
        with self.assertRaises(ValueError):
            config_type({"expert_offload": True, "exclusive_load_pool_enabled": True})
        with self.assertRaises(TypeError):
            config_type(valid | {"exclusive_load_pool_enabled": "true"})
        for size in (True, "1024", 1.5, None):
            with self.subTest(size=size), self.assertRaises(TypeError):
                config_type(valid | {"exclusive_load_pool_size_mb": size})
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                config_type(valid | {"exclusive_load_pool_size_mb": size})

    def test_host_source_tracks_rotating_owner(self):
        manager = SimpleNamespace(_exclusive_pool=self.pool)
        source = method("_expert_src_storage")
        original = source(manager, 0, 2, "w13")
        self.exchange(0, 2)
        with self.assertRaises(KeyError):
            source(manager, 0, 2, "w13")
        self.exchange(1, 3)
        self.exchange(0, 0)
        self.assertIsNot(source(manager, 0, 2, "w13"), original)
        self.assertEqual(source(manager, 0, 2, "w13").value, bundle(2)["w13"].value)

    def test_initial_allocation_skips_resident_weights_and_metadata(self):
        allocator = Mock(side_effect=lambda shape, dtype: Buffer(0))
        manager = SimpleNamespace(
            offload_config=SimpleNamespace(exclusive_expert_storage=True),
            num_device_experts_for_layer=lambda idx: 2,
            _allocate_expert_host_tensor=allocator)
        allocate = method("_allocate_layer_host_experts")
        slots = allocate(manager, (16, 16), "uint8", 6, 0)
        self.assertEqual(slots[:2], [None, None])
        self.assertEqual(allocator.call_count, 4)
        self.assertTrue(all(t is not None for t in slots[2:]))
        allocator.reset_mock()
        manager.offload_config.exclusive_expert_storage = False
        allocate(manager, (16, 16), "uint8", 6, 0)
        self.assertEqual(allocator.call_count, 6)

    def test_sparse_startup_never_requires_resident_host_bundles(self):
        pool = ExclusiveExpertPool()
        allocate = Mock(side_effect=lambda template: bundle(0))
        for layer in range(3):
            pool.add_layer(layer, [None, None] + [bundle(i) for i in range(2, 6)],
                           [0, 1, -1, -1, -1, -1], allocate_spare=allocate)
        self.assertEqual(len(pool.host), 12)
        self.assertEqual(len(pool.spares), 1)
        allocate.assert_called_once()
        self.assertEqual(len(pool.resident), 6)

    def test_format_conversion_never_batches_a_full_exclusive_layer(self):
        manager = SimpleNamespace(
            offload_config=SimpleNamespace(exclusive_expert_storage=True),
            w13_weights_cpu=[[None, None, object(), object()], [None, object(), object(), object()]],
            _cast_cpu_expert_batch=Mock())
        convert = method("_cast_cpu_weights_to_device_format")
        convert(manager, mxfp4=True)
        calls = [call.args[:2] for call in manager._cast_cpu_expert_batch.call_args_list]
        self.assertEqual(calls, [(0, [2]), (0, [3]), (1, [1]), (1, [2]), (1, [3])])
        manager.offload_config.exclusive_expert_storage = False
        manager._cast_cpu_expert_batch.reset_mock()
        convert(manager)
        self.assertEqual(manager._cast_cpu_expert_batch.call_count, 2)

    def test_resident_weight_loader_never_reads_or_clones_checkpoint_source(self):
        manager = SimpleNamespace(_weight_load_calls=0, native_mxfp4_enabled=False,
                                  _shard_local=lambda eid: eid, w13_weights_cpu=[[None]],
                                  w2_weights_cpu=[[None]])
        source = Mock()
        method("load_w13")(manager, 0, 0, source, "w1")
        method("load_w2")(manager, 0, 0, source)
        self.assertEqual(source.mock_calls, [])

    def test_cold_weight_loader_is_synchronous_without_owned_clones(self):
        destination = SimpleNamespace(shape=(4, 4))
        manager = SimpleNamespace(_weight_load_calls=0, native_mxfp4_enabled=False,
                                  offload_config=SimpleNamespace(exclusive_expert_storage=True,
                                                                exclusive_load_pool_enabled=False),
                                  _load_phase_start=0.0,
                                  _shard_local=lambda eid: eid, w13_weights_cpu=[[destination]],
                                  w2_weights_cpu=[[destination]], _copy_w13_shard=Mock(), _copy_w2=Mock())
        copy = method("_copy_exclusive_loaded_shard", time=time)
        manager._copy_exclusive_loaded_shard = lambda *args: copy(manager, *args)
        source = SimpleNamespace(ndim=2, shape=(2, 4))  # No cpu()/clone() methods.
        method("load_w13")(manager, 0, 0, source, "w3")
        method("load_w2")(manager, 0, 0, source)
        manager._copy_w13_shard.assert_called_once_with(destination, source, "w3", 2)
        manager._copy_w2.assert_called_once_with(destination, source)
        self.assertGreater(manager._load_phase_start, 0)

    def test_prefill_initialization_does_not_require_complete_host_weights(self):
        transport = Mock()
        manager = SimpleNamespace(
            _exclusive_pool=self.pool, offload_config=SimpleNamespace(shard_per_rank=False),
            w13_weights_cpu=[[None] * 6], load_stream=object(),
            _get_h2d_transport=lambda: transport,
            _build_prefill_local_d2d_tasks=lambda *args: [("device", args[-1])],
            _build_prefill_h2d_tasks=lambda layer, slot, eid, dest: [("host", eid)],
            _refresh_prefill_fp32_scale=Mock(), _synchronize_h2d=Mock())
        init = method("_init_prefill_pool_data", torch_npu=SimpleNamespace(npu=SimpleNamespace(stream=lambda _: nullcontext())))
        init(manager, None, 6, 2)
        expected = [("device" if i < 2 else "host", i) for i in range(6)]
        self.assertEqual(transport.copy_batch.call_count, 2)
        for call in transport.copy_batch.call_args_list:
            self.assertEqual(call.args[0], expected)

    def test_scale_shards_write_final_slices_in_either_order(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Optional NumPy needed for numerical scale-layout reference")

        class Tensor:
            def __init__(self, data):
                self.data = data
                self.shape = data.shape
                self.dtype = data.dtype

            def reshape(self, *shape):
                return Tensor(self.data.reshape(*shape))

            def view(self, dtype):
                return Tensor(self.data.view(dtype))

            def transpose(self, a, b):
                return Tensor(self.data.swapaxes(a, b))

            def narrow(self, dim, offset, size):
                slices = [slice(None)] * self.data.ndim
                slices[dim] = slice(offset, offset + size)
                return Tensor(self.data[tuple(slices)])

            def numel(self):
                return self.data.size

            def copy_(self, source):
                self.data[...] = source.data

        # Execute the real MXFP4 layout helper using a NumPy tensor adapter.
        layout_path = ROOT / "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py"
        node = next(n for n in ast.parse(layout_path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "apply_mxfp4_weight_scale_layout")
        namespace = {"torch": SimpleNamespace(Tensor=Tensor, uint8=np.dtype("uint8"))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(layout_path), "exec"), namespace)
        layout = namespace["apply_mxfp4_weight_scale_layout"]
        fake_module = SimpleNamespace(apply_mxfp4_weight_scale_layout=layout)
        key = "vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4"
        copy = method("_copy_exclusive_scale_shard", torch=namespace["torch"])
        for dtype in (np.uint8, np.float32):
            w1 = Tensor(np.arange(24, dtype=dtype).reshape(3, 8))
            w3 = Tensor(np.arange(24, 48, dtype=dtype).reshape(3, 8))
            combined = Tensor(np.concatenate((w1.data, w3.data)))
            expected = layout(combined).data if dtype == np.uint8 else combined.data.reshape(-1)
            for order in ((("w1", w1), ("w3", w3)), (("w3", w3), ("w1", w1))):
                target = Tensor(np.zeros(expected.shape, dtype=dtype))
                with patch.dict(sys.modules, {key: fake_module}):
                    for shard, source in order:
                        copy(target, source, shard, "w13_weight_scale")
                np.testing.assert_array_equal(target.data, expected)

    def test_mxfp4_finalization_preserves_already_nz_slots_without_native_cpu(self):
        # The DeepSeek loader already produced NZ. Touching .data would start
        # the erroneous transpose/contiguous/recast sequence from the report.
        class AlreadyNZ:
            @property
            def data(self):
                raise AssertionError("Already-NZ storage must be preserved verbatim")

        layer = SimpleNamespace(w13_weight=AlreadyNZ(), w2_weight=AlreadyNZ())
        npu = SimpleNamespace(get_npu_format=Mock(), npu_format_cast=Mock())
        manager = SimpleNamespace(moe_layers=[layer], native_mxfp4_enabled=False,
                                  offload_config=SimpleNamespace(exclusive_expert_storage=True))
        convert = method("_cast_device_slots_to_mxfp4_nz", torch_npu=npu, _expert_weight=getattr)
        class Format(Enum):
            FRACTAL_NZ_C0_16 = "FRACTAL_NZ_C0_16"

        # Plain Enum deliberately does not compare equal to integer 29.
        for source_format in (29, "FRACTAL_NZ", "FRACTAL_NZ_C0_16",
                              Format.FRACTAL_NZ_C0_16, "Format.FRACTAL_NZ_C0_16"):
            with self.subTest(source_format=source_format):
                npu.get_npu_format.reset_mock(return_value=True)
                npu.get_npu_format.return_value = source_format
                for _ in range(2):
                    convert(manager)
                self.assertEqual(npu.get_npu_format.call_count, 4)
        npu.npu_format_cast.assert_not_called()

    def test_mxfp4_mixed_layout_converts_only_base_format_weight(self):
        nz, base, converted = Mock(), Mock(), Mock()
        npu = SimpleNamespace(get_npu_format=lambda w: 29 if w is nz else 2,
                              npu_format_cast=Mock(return_value=converted), float4_e2m1fn_x2="fp4")
        layer = SimpleNamespace(w13_weight=nz, w2_weight=base)
        manager = SimpleNamespace(moe_layers=[layer], native_mxfp4_enabled=False,
                                  offload_config=SimpleNamespace(exclusive_expert_storage=True))
        convert = method("_cast_device_slots_to_mxfp4_nz", torch_npu=npu, _expert_weight=getattr,
                         torch=SimpleNamespace(uint8="uint8", float8_e4m3fn="fp8"))
        original_nz_data = nz.data
        convert(manager)
        npu.npu_format_cast.assert_called_once()
        self.assertIs(nz.data, original_nz_data)
        self.assertIs(base.data, converted.transpose.return_value)
        converted.transpose.assert_called_once_with(1, 2)

    def test_mxfp4_base_format_failure_is_not_silently_ignored(self):
        base = Mock(shape=(2, 32, 16), dtype="uint8")
        original = base.data
        npu = SimpleNamespace(get_npu_format=Mock(return_value=2),
                              npu_format_cast=Mock(side_effect=RuntimeError("ACL 361001")),
                              float4_e2m1fn_x2="fp4")
        manager = SimpleNamespace(moe_layers=[SimpleNamespace(w13_weight=base)],
                                  offload_config=SimpleNamespace(exclusive_expert_storage=True))
        convert = method("_cast_device_slots_to_mxfp4_nz", torch_npu=npu, _expert_weight=getattr,
                         torch=SimpleNamespace(uint8="uint8", float8_e4m3fn="fp8"))
        with self.assertRaisesRegex(RuntimeError, "weight=w13_weight, source_format=2") as error:
            convert(manager)
        self.assertEqual(str(error.exception.__cause__), "ACL 361001")
        self.assertIs(base.data, original)

    def test_mxfp4_other_named_formats_are_not_treated_as_nz_c0_16(self):
        base = Mock(shape=(2, 32, 16), dtype="uint8")
        npu = SimpleNamespace(get_npu_format=Mock(),
                              npu_format_cast=Mock(side_effect=RuntimeError("cast failed")),
                              float4_e2m1fn_x2="fp4")
        manager = SimpleNamespace(moe_layers=[SimpleNamespace(w13_weight=base)],
                                  offload_config=SimpleNamespace(exclusive_expert_storage=True))
        convert = method("_cast_device_slots_to_mxfp4_nz", torch_npu=npu, _expert_weight=getattr,
                         torch=SimpleNamespace(uint8="uint8", float8_e4m3fn="fp8"))
        # Do not accept arbitrary FRACTAL_NZ_* names with different blocking.
        for source_format in ("ND", "FRACTAL_NZ_C0_32", "UNKNOWN"):
            with self.subTest(source_format=source_format):
                npu.get_npu_format.return_value = source_format
                npu.npu_format_cast.reset_mock()
                with self.assertRaisesRegex(RuntimeError, f"source_format={source_format}"):
                    convert(manager)
                npu.npu_format_cast.assert_called_once()


class ExclusiveLoadPoolTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Optional NumPy needed for loader tensor adapters")
        self.np = np

        class Tensor:
            def __init__(self, data):
                self.data = data
                self.shape, self.dtype, self.ndim = data.shape, data.dtype, data.ndim

            def numel(self):
                return self.data.size

            def element_size(self):
                return self.data.itemsize

            def narrow(self, dim, offset, size):
                slices = [slice(None)] * self.ndim
                slices[dim] = slice(offset, offset + size)
                return Tensor(self.data[tuple(slices)])

            def view(self, dtype):
                return Tensor(self.data.view(dtype))

            def reshape(self, *shape):
                return Tensor(self.data.reshape(*shape))

            def t(self):
                return Tensor(self.data.T)

            def __getitem__(self, index):
                return Tensor(self.data[index])

            def copy_(self, source):
                self.data[...] = source.data

        self.Tensor = Tensor
        self.threads = [16]
        self.torch = SimpleNamespace(
            uint8=np.uint8, get_num_threads=lambda: self.threads[-1], set_num_threads=self.threads.append,
            empty=Mock(side_effect=lambda size, dtype, device: Tensor(np.empty(size, dtype=dtype))))

    def manager(self, enabled):
        manager = SimpleNamespace(
            offload_config=SimpleNamespace(exclusive_expert_storage=True, exclusive_load_pool_enabled=enabled,
                                          exclusive_load_pool_size_mb=1),
            _load_pool=None, _exclusive_load_pool=None, _load_phase_start=0.0, _saved_num_threads=None,
            _LOAD_POOL_WORKERS=2, _LOAD_POOL_DRAIN_EVERY=128, _load_futures=[],
            _weight_load_calls=0, _weight_load_secs=0.0, native_mxfp4_enabled=False,
            _shard_local=lambda eid: eid,
        )
        namespace = dict(torch=self.torch, time=time, logger=Mock(),
                         LoadBufferPool=LoadBufferPool, ThreadPoolExecutor=ThreadPoolExecutor)
        for name in ("_get_load_pool", "_copy_exclusive_loaded_shard", "_drain_futures", "drain_load_pool",
                     "load_w13", "load_w2", "_load_scale_shard"):
            function = method(name, **namespace)
            setattr(manager, name, lambda *args, fn=function: fn(manager, *args))
        for name in ("_copy_w13_shard", "_copy_w2", "_copy_exclusive_scale_shard"):
            setattr(manager, name, method(name, **namespace))
        self.addCleanup(manager.drain_load_pool)
        return manager

    def test_enabled_and_disabled_loaders_produce_identical_weights_and_scales(self):
        np, Tensor = self.np, self.Tensor
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                manager = self.manager(enabled)
                w13 = Tensor(np.zeros((4, 6), dtype=np.uint8))
                w2 = Tensor(np.zeros((3, 4), dtype=np.uint8))
                scales = Tensor(np.zeros(6, dtype=np.float32))
                manager.w13_weights_cpu = [[None, w13]]
                manager.w2_weights_cpu = [[None, w2]]
                manager.scale_cpu_buffers = {"w13_weight_scale": [[None, scales]]}
                # Resident sources must not even be inspected or start a pool.
                for shard in ("w1", "w3"):
                    manager.load_w13(0, 0, object(), shard)
                    manager._load_scale_shard(0, 0, "w13_weight_scale", shard, object())
                manager.load_w2(0, 0, object())
                self.assertIsNone(manager._load_pool)
                expected = []
                for shard in ("w3", "w1"):
                    data = np.arange(12, dtype=np.uint8).reshape(3, 4) + (20 if shard == "w3" else 0)
                    expected.append(data.T.copy())
                    manager.load_w13(0, 1, Tensor(data), shard)
                    data.fill(255)  # Loader source lifetime ends at return.
                    scale = np.arange(3, dtype=np.float32) + (10 if shard == "w3" else 0)
                    manager._load_scale_shard(0, 1, "w13_weight_scale", shard, Tensor(scale))
                    scale.fill(255)
                data = np.arange(12, dtype=np.uint8).reshape(4, 3)
                manager.load_w2(0, 1, Tensor(data))
                data.fill(255)
                pool = manager._exclusive_load_pool
                manager.drain_load_pool()
                np.testing.assert_array_equal(w13.data, np.concatenate(expected[::-1], axis=1))
                np.testing.assert_array_equal(w2.data, np.arange(12, dtype=np.uint8).reshape(4, 3).T)
                np.testing.assert_array_equal(scales.data, [0, 1, 2, 10, 11, 12])
                self.assertGreater(manager._weight_load_secs, 0)
                self.assertIsNone(manager._exclusive_load_pool)
                self.assertIsNone(manager._load_pool)
                if enabled:
                    self.assertEqual(pool.submitted, 5)
                    self.assertIsNone(pool.storage)
                    self.torch.empty.assert_called_once_with(1024 * 1024, dtype=np.uint8, device="cpu")
                    self.assertEqual(self.threads, [16, 1, 16])
                else:
                    self.torch.empty.assert_not_called()
                    self.assertEqual(self.threads, [16])

    def test_worker_failure_drains_pool_and_restores_torch_threads(self):
        manager = self.manager(True)
        source = self.Tensor(self.np.zeros(16, dtype=self.np.uint8))
        manager._copy_exclusive_loaded_shard(object(), source, Mock(side_effect=RuntimeError("copy failed")))
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            manager.drain_load_pool()
        self.assertIsNone(manager._exclusive_load_pool)
        self.assertIsNone(manager._load_pool)
        self.assertEqual(self.threads, [16, 1, 16])

    def test_oversized_shard_copies_synchronously_without_arena(self):
        manager = self.manager(True)
        source = self.Tensor(self.np.zeros(1024 * 1024 + 1, dtype=self.np.uint8))
        copy = Mock()
        target = object()
        manager._copy_exclusive_loaded_shard(target, source, copy)
        copy.assert_called_once_with(target, source)
        self.torch.empty.assert_not_called()
        self.assertEqual(manager._exclusive_load_pool.synchronous, 1)
        manager.drain_load_pool()
        self.assertEqual(self.threads, [16, 1, 16])

    def test_staging_failure_releases_arena_and_restores_threads(self):
        manager = self.manager(True)
        source = self.Tensor(self.np.zeros(16, dtype=self.np.uint8))
        copy = Mock()
        with patch.object(self.Tensor, "copy_", side_effect=RuntimeError("staging failed")):
            with self.assertRaisesRegex(RuntimeError, "staging failed"):
                manager._copy_exclusive_loaded_shard(object(), source, copy)
        copy.assert_not_called()
        self.assertIsNone(manager._exclusive_load_pool)
        self.assertIsNone(manager._load_pool)
        self.assertEqual(self.threads, [16, 1, 16])


if __name__ == "__main__":
    unittest.main()
