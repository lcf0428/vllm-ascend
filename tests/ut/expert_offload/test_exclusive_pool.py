"""Dependency-free ownership tests; also runnable directly with unittest.

Manager adapters are extracted from their actual AST to test them without
importing vLLM/torch_npu. This checks orchestration, not accelerator kernels.
"""

import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/expert_offload"
spec = importlib.util.spec_from_file_location("exclusive_pool", SOURCE / "exclusive_pool.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ExclusiveExpertPool = module.ExclusiveExpertPool


def method(name, **namespace):
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
            events.append("d2h" if hasattr(batch[0], "direction") else "h2d")
            for item in batch:
                item.destination.copy_(item.source)
        manager = SimpleNamespace(
            _exclusive_pool=self.pool, _build_expert_h2d_tasks=tasks,
            _get_h2d_transport=lambda: SimpleNamespace(copy_batch=copy_batch),
            _synchronize_h2d=lambda: events.append("sync"),
            _refresh_expert_fp32_scale=lambda *args: events.append("refresh"))
        swap = method("_exclusive_swap", H2DCopyTask=task, CopyDirection=SimpleNamespace(D2H="d2h"))
        for eid in (2, 0, 4, 2):
            events.clear()
            swap(manager, object(), 0, 0, eid)
            self.assertEqual(events, ["d2h", "sync", "h2d", "sync", "refresh"])
            for name, tensor in self.device[0].items():
                self.assertEqual(tensor.value, bundle(eid)[name].value)

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
        valid = {"expert_offload": True, "exclusive_expert_storage": True}
        self.assertTrue(config_type(valid).prefill_delta_load)
        for invalid in ({"cpu_moe": {"enabled": True}}, {"enable_multi_card": True},
                        {"expert_prefetch_enabled": True}, {"h2d_backend": "memfabric"},
                        {"prefill_split_gmm": True}, {"expert_offload": False}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config_type(valid | invalid)
        with self.assertRaises(TypeError):
            config_type(valid | {"exclusive_expert_storage": "true"})

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
                                  offload_config=SimpleNamespace(exclusive_expert_storage=True),
                                  _shard_local=lambda eid: eid, w13_weights_cpu=[[destination]],
                                  w2_weights_cpu=[[destination]], _copy_w13_shard=Mock(), _copy_w2=Mock())
        source = SimpleNamespace(ndim=2, shape=(2, 4))  # No cpu()/clone() methods.
        method("load_w13")(manager, 0, 0, source, "w3")
        method("load_w2")(manager, 0, 0, source)
        manager._copy_w13_shard.assert_called_once_with(destination, source, "w3", 2)
        manager._copy_w2.assert_called_once_with(destination, source)

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
        npu = SimpleNamespace(get_npu_format=Mock(return_value=29), npu_format_cast=Mock())
        manager = SimpleNamespace(moe_layers=[layer], native_mxfp4_enabled=False,
                                  offload_config=SimpleNamespace(exclusive_expert_storage=True))
        convert = method("_cast_device_slots_to_mxfp4_nz", torch_npu=npu, _expert_weight=getattr)
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


if __name__ == "__main__":
    unittest.main()
