import ctypes
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from vllm_ascend.expert_offload import cpu_moe_calibration as calibration
from vllm_ascend.expert_offload.host_transfer_policy import HostTransferPolicy


def test_cpu_microbenchmark_runs_single_and_repeated_routes(monkeypatch):
    calls = []
    clock = 0

    class UnpinnedTorch:
        def __getattr__(self, name):
            value = getattr(torch, name)
            if name not in ("empty", "full", "zeros"):
                return value

            def allocate(*args, pin_memory=False, **kwargs):
                return value(*args, **kwargs)

            return allocate

    class Probe:
        def __init__(self, pool, layer, hidden, intermediate, topk, capacity, pointers):
            assert capacity == 2
            assert pointers == [[17]]

        def forward_task(self, count, k, ids, weights, hidden, output, incremental):
            return count, k, ids

    def submit(task):
        nonlocal clock
        count, k, ids = task
        tokens = ctypes.c_int32.from_address(count).value
        routes = (ctypes.c_int64 * (tokens * k)).from_address(ids)
        assert all(routes[row * k] == routes[0] and routes[row * k + 1] == -1 for row in range(tokens))
        calls.append(tokens)
        clock += 100_000 if tokens == 1 else 120_000

    fake_kt = SimpleNamespace(kt_kernel_ext=SimpleNamespace(moe=SimpleNamespace(NZMOE=Probe)))
    monkeypatch.setitem(sys.modules, "kt_kernel", fake_kt)
    monkeypatch.setattr(calibration, "torch", UnpinnedTorch())
    monkeypatch.setattr(calibration, "perf_counter_ns", lambda: clock)
    backend = SimpleNamespace(
        config=SimpleNamespace(layer_idx=0, hidden_size=32, intermediate_size=32, top_k=2),
        _wrapper=SimpleNamespace(
            cpu_infer=SimpleNamespace(backend_=object(), submit=submit, sync=lambda _: None),
            moe=SimpleNamespace(weight_pointers=lambda: [[17]]),
        ),
    )

    assert calibration._measure_cpu(backend, expert_count=4) == (100_000, 120_000)
    assert calls.count(1) == calls.count(2) == calibration._SAMPLES + 2


def test_copy_microbenchmark_keeps_resident_destinations_untouched(monkeypatch):
    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self, stream):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 0.15

    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=Event, stream=lambda _: nullcontext()), raising=False)
    sources = [torch.full((8,), i + 1, dtype=torch.uint8) for i in range(4)]
    resident = torch.zeros(8, dtype=torch.uint8)
    plan = SimpleNamespace(
        fields=[
            ((source.untyped_storage(),), (resident.untyped_storage(),), (8,), name)
            for source, name in zip(sources, ("w13", "w2", "scale13", "scale2"))
        ],
    )

    def copy_batch(tasks):
        assert [task.nbytes for task in tasks] == [8, 8, 8, 8]
        for task, source in zip(tasks, sources):
            assert task.source.data_ptr() == source.data_ptr()
            assert task.destination.data_ptr() != resident.data_ptr()
            task.destination.copy_(task.source)

    transport = SimpleNamespace(copy_batch=copy_batch)
    assert calibration._measure_copy(plan, transport, object(), "cpu", expert_count=1) == 150_000
    assert torch.count_nonzero(resident) == 0


def test_duplicate_and_independent_routes_have_different_costs():
    costs = calibration.CpuMoeCosts(100_000, 120_000, 150_000)
    planner = HostTransferPolicy(
        experts=4, slots=2, config=SimpleNamespace(balance_max_transfers=2), costs=costs,
    )
    assert planner.plan([[2], [2]], [[1.0], [1.0]], [-1] * 4).loads == ()
    assert planner.plan([[2], [3]], [[1.0], [1.0]], [-1] * 4).loads == ((0, 2),)
