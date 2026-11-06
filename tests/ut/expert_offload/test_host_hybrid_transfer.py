"""Host admission, route conservation, and independent completion boundaries."""

import importlib
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.expert_offload.host_transfer_policy import HostTransferPolicy
from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy


def config(policy="load_balance", **overrides):
    values = dict(transfer_policy=policy, balance_cpu_expert_us=70.0, balance_copy_us=450.0, balance_max_transfers=32)
    values.update(overrides)
    return SimpleNamespace(**values)


def planner(experts=16, slots=4, **kwargs):
    lrc = LRCExpertCachePolicy(num_layers=1, num_experts=experts, cache_size=slots, topk=4)
    return HostTransferPolicy(experts=experts, slots=slots, config=config(**kwargs), lrc=lrc)


def test_all_hits_and_no_legal_victim():
    p = planner(experts=8, slots=2)
    result = p.plan([[0, 1]], [[0.6, 0.4]], [0, 1] + [-1] * 6)
    assert not result.loads and result.cpu_ids == ((-1, -1),)
    result = p.plan([[0, 1, 2, 3]], [[0.25] * 4], [0, 1] + [-1] * 6)
    assert not result.loads and result.cpu_ids == ((-1, -1, 2, 3),)


def test_lrc_victims_protect_hits_and_conserve_every_route():
    p = planner(experts=16, slots=4, balance_cpu_expert_us=1.0, balance_copy_us=1.0)
    p.lrc.observe(0, [[2]] * 3)  # retain expert 2 over cold 1 and 3
    ids = [[0, 4, 5, 6, 7, 8, 9]]
    weights = [[0.1] * 7]
    original = [0, 1, 2, 3] + [-1] * 12
    result = p.plan(ids, weights, original)
    assert len(result.loads) == 3
    assert result.mapping[0] == 0  # active NPU hit must survive
    for row, ws, cpu_row, cpu_ws in zip(ids, weights, result.cpu_ids, result.cpu_weights):
        for eid, weight, cpu_eid, cpu_weight in zip(row, ws, cpu_row, cpu_ws):
            npu = result.mapping[eid] >= 0
            assert npu != (cpu_eid >= 0)
            assert (weight if npu else 0.0) + cpu_weight == weight
    assert original == [0, 1, 2, 3] + [-1] * 12  # plan does not publish live state


def test_pruned_routes_do_not_update_lrc_or_get_computed():
    p = planner(experts=8, slots=2, policy="first_miss")
    result = p.plan([[2, 3, -1, 99]], [[0.0, 1.0, 1.0, 1.0]], [0, 1] + [-1] * 6)
    assert [e for _, e in result.loads] == [3]
    assert result.cpu_ids == ((-1, -1, -1, -1),)
    assert p.lrc.layer_states[0].freq[2] == 0


def test_free_slots_are_used_before_lrc_eviction():
    p = planner(experts=8, slots=3, policy="first_miss")
    result = p.plan([[3]], [[1.0]], [0, -1, -1, -1, -1, -1, -1, -1])
    assert result.loads == ((1, 3),) and result.mapping[0] == 0


@pytest.mark.parametrize("bad", [[0, 0, -1, -1], [2, -1, -1, -1], [-2, -1, -1, -1]])
def test_invalid_mapping_is_rejected(bad):
    with pytest.raises(ValueError, match="slot"):
        planner(experts=4, slots=2).plan([[0]], [[1.0]], bad)


def test_consecutive_runtime_plans_observe_new_residency():
    p = planner(experts=8, slots=2, policy="first_miss")
    first = p.plan([[2]], [[1.0]], [0, 1] + [-1] * 6)
    second = p.plan([[2, 3]], [[0.5, 0.5]], first.mapping)
    assert second.mapping[2] == first.mapping[2]
    assert [e for _, e in second.loads] == [3]
    assert p.lrc.layer_step(0) == 2


def test_cpp_lrc_matches_python_over_repeated_plans():
    pytest.importorskip("_vllm_ascend_lrc")
    import random

    from vllm_ascend.expert_offload.lrc_policy import CppLRCExpertCachePolicy

    rng = random.Random(21)
    py = planner(experts=16, slots=4, policy="first_miss")
    native = HostTransferPolicy(
        experts=16,
        slots=4,
        config=py.config,
        lrc=CppLRCExpertCachePolicy(num_layers=1, num_experts=16, cache_size=4, topk=4),
    )
    mapping = [0, 1, 2, 3] + [-1] * 12
    for _ in range(100):
        ids = [[rng.randrange(16) for _ in range(4)] for _ in range(3)]
        weights = [[rng.random() for _ in row] for row in ids]
        a, b = py.plan(ids, weights, mapping), native.plan(ids, weights, mapping)
        assert a == b
        mapping = a.mapping


@pytest.fixture
def runtime(monkeypatch):
    npu = SimpleNamespace(stream=lambda _: nullcontext())
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    return importlib.import_module("vllm_ascend.expert_offload.host_transfer")


@pytest.mark.parametrize("first", ["cpu", "weights"])
def test_output_and_weight_completion_have_no_ordering_dependency(runtime, first):
    from vllm_ascend.expert_offload.cpu_backend.async_completion import (
        CompletionWorker,
        DrainRequest,
        SubmissionRequest,
    )

    gates = {name: threading.Event() for name in ("cpu", "weights")}
    entered = {name: threading.Event() for name in gates}
    ready = {name: threading.Event() for name in gates}
    operations, errors = [], []

    def wait(name):
        entered[name].set()
        assert gates[name].wait(3)

    def event(name):
        def record(_):
            operations.append(name + "_ready")
            ready[name].set()

        return SimpleNamespace(record=record)

    dma = runtime.HostWeightWorker.__new__(runtime.HostWeightWorker)
    dma.stream = object()
    dma.manager = SimpleNamespace(_load_expert_weights_into_slots=lambda *_: wait("weights"))
    mapping = SimpleNamespace(copy_=lambda *_args, **_kw: operations.append("mapping"))
    transfer = SimpleNamespace(layer=object(), layer_idx=0)
    weight_job = runtime.WeightRequest(transfer, object(), mapping, (), event("weights"))
    completion = CompletionWorker.__new__(CompletionWorker)
    completion.stream = object()
    completion._release_backend = lambda _: None
    backend = SimpleNamespace(_wrapper=SimpleNamespace(drain_pinned_forward=lambda: wait("cpu")))
    cpu_job = DrainRequest(SubmissionRequest(backend, object(), 0, event("cpu")), 1, False, 0)

    def run(fn, arg):
        try:
            fn(arg)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(dma._process, weight_job)),
        threading.Thread(target=run, args=(lambda j: completion._drain_one(j, None), cpu_job)),
    ]
    for thread in threads:
        thread.start()
    try:
        assert all(e.wait(1) for e in entered.values())
        second = "weights" if first == "cpu" else "cpu"
        gates[first].set()
        assert ready[first].wait(1)
        assert not ready[second].is_set()  # first completion is not blocked by the other
        gates[second].set()
        assert ready[second].wait(1)
    finally:
        for gate in gates.values():
            gate.set()
        for thread in threads:
            thread.join(3)
    assert not errors and not any(t.is_alive() for t in threads)
    assert operations.index("mapping") < operations.index("weights_ready")


def test_dma_failure_never_publishes_weight_ready(runtime):
    worker = runtime.HostWeightWorker.__new__(runtime.HostWeightWorker)
    worker.stream = object()

    def fail(*_):
        raise RuntimeError("copy failed")

    worker.manager = SimpleNamespace(_load_expert_weights_into_slots=fail)
    signals = []
    request = runtime.WeightRequest(
        SimpleNamespace(layer=object(), layer_idx=0),
        object(),
        object(),
        (),
        SimpleNamespace(record=lambda _: signals.append("ready")),
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        worker._process(request)
    assert not signals


def test_graph_capture_defers_planning_and_keeps_weight_output_waits_separate(monkeypatch):
    from vllm_ascend.expert_offload.cpu_backend import kt_kernel_backend as kt
    from vllm_ascend.expert_offload.cpu_backend import poll_submission

    operations, callbacks, contexts = [], [], []

    class Event:
        npu_event = 1

        def reset(self, stream):
            operations.append(("reset", self))

    class Stream:
        def record_event(self, event):
            operations.append(("record", event))

        def wait_event(self, event):
            operations.append(("wait", event))

    main, inputs = Stream(), Stream()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=Event, stream=lambda _: nullcontext()), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=SimpleNamespace(ExternalEvent=Event)))
    monkeypatch.setattr(kt, "_current_stream", lambda _: main)
    monkeypatch.setattr(kt, "_shared_cpu_graph_stream", lambda _: inputs)

    def context(hidden):
        ready = Event()
        ctx = SimpleNamespace(
            ready=ready, transfer=SimpleNamespace(worker=SimpleNamespace(raise_if_failed=lambda: None))
        )
        ctx.stage = lambda stream: (ready.reset(stream), operations.append(("mapping_d2h", ctx)))
        ctx.prepare = lambda: operations.append(("runtime_plan", ctx))
        contexts.append(ctx)
        return ctx

    def publish(backend, hidden, stream, ready, before_submit=None):
        operations.append(("publish", ready))
        callbacks.append(before_submit)

    monkeypatch.setattr(poll_submission, "record_submission", publish)
    backend = kt.KtKernelCPUExpertBackend.__new__(kt.KtKernelCPUExpertBackend)
    backend._require_graph_methods = lambda: None
    backend._pin_graph_batch_size = lambda _: None
    backend._poll_sequence_slots = []
    backend._graph_tasks = []
    backend._host_transfer = SimpleNamespace(context=context)
    backend._wrapper = SimpleNamespace(
        copy_inputs_to_cpu_buffers=lambda *_: operations.append(("inputs_d2h", None)),
        get_forward_output_buffers=lambda _: (torch.zeros(1, 4), None),
    )
    for _ in range(2):
        task = backend._submit_graph(torch.zeros(1, 4), torch.zeros(1, 2, dtype=torch.int64), torch.ones(1, 2))
        assert not any(op == "runtime_plan" for op, _ in operations)
        output = task._output_ready
        assert operations.index(("reset", output)) < operations.index(("publish", output))
        assert operations.index(("reset", contexts[-1].ready)) < operations.index(("publish", output))
        operations.clear()
        task.wait_for_weights()
        assert operations == [("wait", task._input_published_event), ("wait", contexts[-1].ready)]
        assert ("wait", output) not in operations
        operations.clear()
        task.wait()
        assert operations[-1] == ("wait", output)
        operations.clear()
    assert contexts[0] is not contexts[1]
    # Simulate repeated notifications: CPU planning runs for each replay,
    # rather than executing once while Python constructs the graph.
    for _ in range(3):
        for callback in callbacks:
            callback()
    assert len(operations) == 6 and all(op == "runtime_plan" for op, _ in operations)


def test_graph_without_poll_library_fails_before_recording_work():
    from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import KtKernelCPUExpertBackend

    backend = KtKernelCPUExpertBackend.__new__(KtKernelCPUExpertBackend)
    backend._require_graph_methods = lambda: None
    with pytest.raises(RuntimeError, match="host callback dispatch has been retired"):
        backend._submit_graph(None, None, None)


def test_host_load_balance_config_and_retired_npu_scheduler():
    import ast
    from pathlib import Path

    path = Path(__file__).parents[3] / "vllm_ascend/ascend_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CpuMoeConfig")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    cls = namespace["CpuMoeConfig"]
    cfg = cls(
        {
            "enabled": True,
            "weight_path": "/unused/layer{layer_idx}.gguf",
            "transfer_scheduler": "cpu",
            "transfer_policy": "load_balance",
        }
    )
    assert cfg.transfer_policy == "load_balance"
    with pytest.raises(ValueError, match="host admission/DMA"):
        cls({"enabled": True, "transfer_scheduler": "npu"})


def test_runtime_planner_masks_pinned_inputs_and_queues_mapping_without_dma_wait(runtime):
    ids = torch.tensor([[0, 2, 3]], dtype=torch.int64)
    weights = torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.bfloat16)
    requests = []
    p = planner(experts=4, slots=2, policy="first_miss")
    transfer = SimpleNamespace(
        worker=SimpleNamespace(raise_if_failed=lambda: None, submit=requests.append),
        backend=SimpleNamespace(_pinned_route_buffers=lambda _: (ids, weights)),
        policy=p,
        layer=SimpleNamespace(log2phy=object()),
    )
    context = runtime.HostTransferContext.__new__(runtime.HostTransferContext)
    context.transfer, context.hidden = transfer, object()
    context.mapping = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    context.ready = object()
    context.prepare()
    assert ids.tolist() == [[-1, -1, 3]]
    assert weights.tolist() == [[0.0, 0.0, 0.25]]
    assert context.mapping.tolist() == [0, -1, 1, -1]
    assert len(requests) == 1 and requests[0].loads == ((1, 2),)
    assert requests[0].ready is context.ready
    assert requests[0].mapping_host is context.mapping
