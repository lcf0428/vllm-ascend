"""Host-cache ownership and delayed-stream regressions, without torch/NPU."""

import importlib.util
import random
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

spec = importlib.util.spec_from_file_location("exclusive_test_helpers", Path(__file__).with_name("test_exclusive_pool.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
ExclusiveExpertPool, bundle, method = helpers.ExclusiveExpertPool, helpers.bundle, helpers.method


class HostPrefetchTests(unittest.TestCase):
    def setUp(self):
        self.pool = ExclusiveExpertPool()
        self.device = {}
        for layer in range(2):
            self.pool.add_layer(layer, [None] * 3 + [bundle(i) for i in range(3, 8)],
                                [0, 1, 2] + [-1] * 5, allocate_spare=lambda template: bundle(0))
            for slot in range(3):
                self.device[layer, slot] = bundle(slot)
        self.pool.configure_shadow_cache(48, lambda template: bundle(0))
        self.queue = []
        self.submissions = []
        self.addresses = self.allocations()

    def allocations(self):
        bundles = (list(self.pool.host.values()) + list(self.pool.spares.values())
                   + list(self.pool.shadows.values()) + [b for values in self.pool.shadow_free.values() for b in values])
        ids = [id(t) for b in bundles for t in b.values()]
        self.assertEqual(len(ids), len(set(ids)), "Two owners share a physical allocation")
        return set(ids)

    def enqueue(self, tasks):
        self.queue.extend(tasks)

    def sync(self):
        while self.queue:
            dst, src, direction = self.queue.pop(0)
            dst.copy_(src)
            self.submissions.append(direction)

    def backup(self, layer, victims, early=True):
        def prepare(plan):
            return [(b[name], self.device[layer, slot][name], "backup")
                    for slot, eid, b in plan for name in b]
        self.pool.prefetch_backups(layer, victims, early, prepare, self.enqueue)

    def swap(self, layer, loads):
        def prepare(plan):
            tasks = []
            for step in plan:
                device = self.device[layer, step.slot]
                if step.download:
                    tasks.extend((step.destination[name], device[name], "d2h") for name in device)
                tasks.extend((device[name], step.source[name], "h2d") for name in device)
            return tasks
        self.pool.swap_batch(layer, loads, prepare, self.enqueue, self.sync)
        self.sync()  # Empty batches need not drain in the pool API.

    def check(self):
        self.assertEqual(self.allocations(), self.addresses)
        self.assertEqual(len(self.pool.host), 10)
        self.assertEqual(len(self.pool.shadows) + sum(map(len, self.pool.shadow_free.values())), 2)
        for key, b in list(self.pool.host.items()) + list(self.pool.shadows.items()):
            for name, tensor in b.items():
                self.assertEqual(tensor.value, bundle(key[1])[name].value)
        for key, eid in self.pool.resident.items():
            self.assertNotIn((key[0], eid), self.pool.host)
            for name, tensor in self.device[key].items():
                self.assertEqual(tensor.value, bundle(eid)[name].value)
        for layer, eid in self.pool.shadows:
            self.assertIn(eid, [e for (idx, _), e in self.pool.resident.items() if idx == layer])

    def test_pending_backup_precedes_overwrite_without_intermediate_wait(self):
        original = dict(self.pool.resident)
        self.backup(1, [0, 1])
        self.assertEqual(self.pool.resident, original)
        self.assertEqual(self.submissions, [])  # Neither D2H completed yet.
        self.swap(1, [(0, 3), (1, 4)])
        self.assertEqual(self.submissions, ["backup"] * 12 + ["h2d"] * 12)
        self.assertEqual(self.pool.shadow_stats["cache_hits"], 2)
        self.check()

    def test_retains_sources_without_cpu_copy_and_hits_on_later_eviction(self):
        source = self.pool.host[0, 3]
        self.swap(0, [(0, 3)])
        self.assertIs(self.pool.shadows[0, 3], source)
        self.submissions.clear()
        self.backup(0, [3], early=False)
        self.swap(0, [(0, 4)])
        self.assertEqual(self.submissions, ["h2d"] * 6)
        self.assertIs(self.pool.host[0, 3], source)
        self.check()

    def test_wrong_prediction_falls_back_without_changing_routing(self):
        self.backup(0, [0])
        self.swap(0, [(1, 5)])  # Actual victim is 1, not predicted 0.
        self.assertEqual(self.submissions, ["backup"] * 6 + ["d2h"] * 6 + ["h2d"] * 6)
        self.assertEqual(self.pool.resident[0, 0], 0)
        self.assertEqual(self.pool.resident[0, 1], 5)
        self.check()

    def test_no_early_d2h_cold_cache_and_nonresident_candidates(self):
        self.backup(0, [0, 0, 7, 999], early=False)
        self.assertFalse(self.queue)
        self.assertEqual(self.pool.shadow_targets, {(0, 0)})
        self.swap(0, [(0, 3)])
        self.assertEqual(self.submissions, ["d2h"] * 6 + ["h2d"] * 6)
        self.check()

    def test_random_predictions_swaps_cross_layer_reuse_and_capacity(self):
        rng = random.Random(731)
        for _ in range(400):
            layer = rng.randrange(2)
            residents = [eid for (idx, _), eid in self.pool.resident.items() if idx == layer]
            self.backup(layer, rng.sample(residents, rng.randrange(4)), early=bool(rng.randrange(2)))
            # Evict/reuse pending backups from a different layer as well.
            target = rng.randrange(2)
            count = rng.randrange(1, 4)
            slots = rng.sample(list(range(3)), count)
            cold = [eid for idx, eid in self.pool.host if idx == target]
            self.swap(target, list(zip(slots, rng.sample(cold, count))))
            self.check()

    def test_backup_submission_failure_poisoning(self):
        original = dict(self.pool.resident)
        with self.assertRaisesRegex(RuntimeError, "DMA"):
            self.pool.prefetch_backups(0, [0], True, lambda plan: plan, Mock(side_effect=RuntimeError("DMA")))
        self.assertTrue(self.pool.failed)
        self.assertFalse(self.pool.shadows)
        self.assertEqual(self.pool.resident, original)

    def test_backup_prepare_failure_does_not_publish_reservations(self):
        enqueue = Mock()
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            self.pool.prefetch_backups(0, [0], True, Mock(side_effect=RuntimeError("prepare")), enqueue)
        enqueue.assert_not_called()
        self.assertFalse(self.pool.failed)
        self.assertFalse(self.pool.shadows)
        self.check()

    def test_budget_too_small_is_rejected(self):
        pool = ExclusiveExpertPool()
        pool.add_layer(0, [bundle(i) for i in range(3)], [0, -1, -1])
        with self.assertRaisesRegex(ValueError, "one expert"):
            pool.configure_shadow_cache(23, lambda template: bundle(0))

    def test_mixed_layout_cache_allocation_stays_within_global_budget(self):
        pool = ExclusiveExpertPool()
        for layer, size in ((0, 4), (1, 8)):
            pool.add_layer(layer, [bundle(i, size) for i in range(4)], [0, -1, -1, -1])
        pool.configure_shadow_cache(73, lambda template: bundle(0, template["w13"].nbytes()))
        self.assertEqual(pool.shadow_capacity_bytes, 72)
        self.assertEqual([len(values) for values in pool.shadow_free.values()], [1, 1])
        pool.prefetch_backups(0, [0], True, lambda plan: plan, lambda tasks: None)
        pool.prefetch_backups(1, [0], True, lambda plan: plan, lambda tasks: None)
        self.assertEqual(pool.shadows[0, 0]["w13"].nbytes(), 4)
        self.assertEqual(pool.shadows[1, 0]["w13"].nbytes(), 8)

    def test_cached_swap_failure_keeps_committed_cache_and_ownership(self):
        self.backup(0, [0])
        self.sync()
        host, residents, shadows = dict(self.pool.host), dict(self.pool.resident), dict(self.pool.shadows)
        with self.assertRaisesRegex(RuntimeError, "completion"):
            self.pool.swap_batch(0, [(0, 3), (1, 4)], lambda plan: plan, lambda tasks: None,
                                 Mock(side_effect=RuntimeError("completion")))
        self.assertEqual(self.pool.host, host)
        self.assertEqual(self.pool.resident, residents)
        self.assertEqual(self.pool.shadows, shadows)
        self.assertTrue(self.pool.failed)

    def test_callback_failure_does_not_resume_graph(self):
        os = Mock()
        callback = method("_exclusive_host_prefetch_callback", logger=Mock(), os=os)
        manager = SimpleNamespace(_exclusive_host_prefetch=Mock(side_effect=RuntimeError("DMA")))
        callback(manager, (1, object(), object()))
        os._exit.assert_called_once_with(1)

    def test_prediction_selects_victims_not_incoming_or_protected_residents(self):
        manager = SimpleNamespace(_exclusive_pool=self.pool, num_total_experts=8, topk=4,
                                  prefetch_topk=1, cache_policy=None,
                                  offload_config=SimpleNamespace(exclusive_host_prefetch_num=2))
        select = method("_select_exclusive_prefetch_victims")
        ids = SimpleNamespace(tolist=lambda: [[1, 5, 6, -1], [1, 5, 99, 6]])
        scores = SimpleNamespace(tolist=lambda: [[1, .5, .3, 0], [1, .4, 0, .2]])
        self.assertEqual(select(manager, 1, ids, scores), [2, 0])
        manager.cache_policy = SimpleNamespace(choose_victims=Mock(return_value=[0, 2]))
        self.assertEqual(select(manager, 1, ids, scores), [0, 2])
        manager.cache_policy.choose_victims.assert_called_once_with(1, {0: 0, 1: 1, 2: 2},
                                                                   protected={1, 5, 6}, count=2)

    def test_host_candidate_limit_is_independent_and_capped_by_model_topk(self):
        manager = SimpleNamespace(_exclusive_pool=self.pool, num_total_experts=8, topk=2,
                                  prefetch_topk=1, cache_policy=None,
                                  offload_config=SimpleNamespace(exclusive_host_prefetch_num=8))
        select = method("_select_exclusive_prefetch_victims")
        ids = SimpleNamespace(tolist=lambda: [[4, 5], [6, 7]])
        scores = SimpleNamespace(tolist=lambda: [[.9, .8], [.7, .6]])
        for legacy_limit in (1, 8):
            manager.prefetch_topk = legacy_limit
            self.assertEqual(select(manager, 1, ids, scores), [2, 1])
        manager.offload_config.exclusive_host_prefetch_num = 1
        self.assertEqual(select(manager, 1, ids, scores), [2])

    def test_prediction_token_limit_preserves_legacy_default_and_overrides_both_routes(self):
        for hash_routing in (False, True):
            with self.subTest(hash_routing=hash_routing):
                hidden, input_ids, table, gate_w = MagicMock(), MagicMock(), MagicMock(), MagicMock()
                hidden.shape, input_ids.shape = (4, 8), (3,)
                linear = Mock()
                linear.return_value.softmax.return_value.topk.return_value = (object(), object())
                torch = SimpleNamespace(full=Mock(), float32="float32")
                predict = method("predict_next_layer_experts_npu", torch=torch,
                                 F=SimpleNamespace(linear=linear),
                                 get_forward_context=lambda: SimpleNamespace(input_ids=input_ids))
                manager = SimpleNamespace(
                    moe_layers=[object(), SimpleNamespace(gate=SimpleNamespace(
                        tid2eid=table if hash_routing else None))],
                    _gate_weights_npu=[None, None if hash_routing else gate_w],
                    prefetch_tokens=1, topk=2)
                # No override preserves legacy behavior. Explicit Host limits
                # remain independent and clamp to available rows.
                for kwargs, expected in (({}, 1), ({"prediction_tokens": 2}, 2),
                                         ({"prediction_tokens": 9}, 3 if hash_routing else 4)):
                    predict(manager, 0, hidden, **kwargs)
                    sampled = input_ids if hash_routing else hidden
                    self.assertEqual(sampled.__getitem__.call_args.args, (slice(None, expected),))
                manager.prefetch_tokens = 9
                predict(manager, 0, hidden, prediction_tokens=2)
                sampled = input_ids if hash_routing else hidden
                self.assertEqual(sampled.__getitem__.call_args.args, (slice(None, 2),))

    def test_manager_submits_only_d2h_without_wait_or_mapping_change(self):
        original = dict(self.pool.resident)
        def task(source, destination, nbytes, **kwargs):
            return SimpleNamespace(source=source, destination=destination, nbytes=nbytes, **kwargs)
        def build(layer, idx, eid, slot, host_bundle=None):
            return [task(t, self.device[idx, slot][name], t.nbytes(), name=name) for name, t in host_bundle.items()]
        submitted = []
        manager = SimpleNamespace(
            _exclusive_pool=self.pool, moe_layers=[object(), object()], load_stream=object(),
            offload_config=SimpleNamespace(exclusive_host_prefetch_d2h=True, cache_stats_log_interval=100),
            _select_exclusive_prefetch_victims=lambda *args: [0, 1], _build_expert_h2d_tasks=build,
            _get_h2d_transport=lambda: SimpleNamespace(copy_batch=submitted.extend), _synchronize_h2d=Mock())
        prefetch = method("_exclusive_host_prefetch", H2DCopyTask=task, CopyDirection=SimpleNamespace(D2H="d2h"),
                          torch=SimpleNamespace(inference_mode=nullcontext),
                          torch_npu=SimpleNamespace(npu=SimpleNamespace(stream=lambda _: nullcontext())), logger=Mock())
        prefetch(manager, (1, object(), object()))
        self.assertEqual(len(submitted), 12)
        self.assertTrue(all(t.direction == "d2h" for t in submitted))
        manager._synchronize_h2d.assert_not_called()
        self.assertEqual(self.pool.resident, original)

    def test_graph_prediction_stages_dedicated_buffers_before_callback(self):
        stream, layer = object(), object()
        ids, weights = MagicMock(), MagicMock()
        ids.shape = (1, 2)
        ids_buffer, weights_buffer = MagicMock(), MagicMock()
        npu = SimpleNamespace(current_stream=lambda: stream, _subscribe_report=Mock(), _launch_host_func=Mock())
        subscribed = set()
        manager = SimpleNamespace(
            _skip_prefill=False, _exclusive_pool=self.pool, offload_threshold=4, moe_layers=[layer, object()],
            offload_config=SimpleNamespace(exclusive_host_prefetch_tokens=2), prefetch_tokens=1,
            _exclusive_prediction_buffers={1: (ids_buffer, weights_buffer)},
            predict_next_layer_experts_npu=Mock(return_value=(weights, ids)),
            _exclusive_host_prefetch_callback=Mock(), _exclusive_host_prefetch=Mock())
        trigger = method("_trigger_exclusive_host_prefetch", _EXTRA_CTX=SimpleNamespace(capturing=True),
                         torch=SimpleNamespace(int32="int32", float32="float32"), torch_npu=SimpleNamespace(npu=npu),
                         get_subscribed_compute_streams=lambda: subscribed)
        hidden = SimpleNamespace(shape=(2, 8))
        trigger(manager, layer, hidden)
        manager.predict_next_layer_experts_npu.assert_called_once_with(0, hidden, prediction_tokens=2)
        npu._subscribe_report.assert_called_once_with(stream)
        npu._launch_host_func.assert_called_once()
        args = npu._launch_host_func.call_args.args
        self.assertIs(args[1], manager._exclusive_host_prefetch_callback)
        self.assertEqual(args[2][0], 1)
        ids_buffer.__getitem__.return_value.copy_.assert_called_once_with(ids.to.return_value, non_blocking=True)
        manager._exclusive_host_prefetch.assert_not_called()
        trigger(manager, layer, SimpleNamespace(shape=(5, 8)))  # Prefill must not launch it.
        self.assertEqual(npu._launch_host_func.call_count, 1)

    def test_manager_cache_hit_omits_all_victim_d2h_fields(self):
        self.backup(0, [0])
        self.sync()
        def task(source, destination, nbytes, **kwargs):
            return SimpleNamespace(source=source, destination=destination, nbytes=nbytes, **kwargs)
        def build(layer, idx, eid, slot, host_bundle=None):
            return [task(t, self.device[idx, slot][name], t.nbytes(), name=name) for name, t in host_bundle.items()]
        submitted = []
        def sync():
            for t in submitted:
                t.destination.copy_(t.source)
        manager = SimpleNamespace(
            _exclusive_pool=self.pool, load_stream=object(), _build_expert_h2d_tasks=build,
            _get_h2d_transport=lambda: SimpleNamespace(copy_batch=submitted.extend),
            _synchronize_h2d=sync, _refresh_expert_fp32_scale=Mock())
        swap = method("_exclusive_swap_batch", H2DCopyTask=task, CopyDirection=SimpleNamespace(D2H="d2h"),
                      torch_npu=SimpleNamespace(npu=SimpleNamespace(stream=lambda _: nullcontext())))
        swap(manager, object(), 0, [(0, 3)])
        self.assertEqual(len(submitted), 6)
        self.assertTrue(all(not hasattr(t, "direction") for t in submitted))
        self.check()


if __name__ == "__main__":
    unittest.main()
