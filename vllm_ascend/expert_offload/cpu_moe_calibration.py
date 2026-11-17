"""Startup microbenchmarks for the shared-NZ CPU MoE cost model."""

from dataclasses import dataclass
from statistics import median
from time import perf_counter_ns

import torch

from .h2d_transfer import H2DCopyTask

_SAMPLES = 9
_SOURCE_EXPERTS = 8


@dataclass(frozen=True)
class CpuMoeCosts:
    first_route_ns: int
    two_routes_ns: int
    copy_expert_ns: int

    @property
    def extra_route_ns(self) -> int:
        return max(0, self.two_routes_ns - self.first_route_ns)


def _measure_cpu(backend, expert_count):
    from kt_kernel import kt_kernel_ext

    config = backend.config
    wrapper = backend._wrapper
    # A separate kernel owner can test two tokens even when serving capacity is one.
    # It borrows the live expert bytes and uses the same CPUInfer pool.
    probe = kt_kernel_ext.moe.NZMOE(
        wrapper.cpu_infer.backend_, config.layer_idx, config.hidden_size,
        config.intermediate_size, config.top_k, 2, wrapper.moe.weight_pointers(),
    )
    hidden = torch.full((2, config.hidden_size), 0.125, dtype=torch.bfloat16, device="cpu", pin_memory=True)
    output = torch.empty(hidden.shape, dtype=hidden.dtype, device="cpu", pin_memory=True)
    ids = torch.full((2, config.top_k), -1, dtype=torch.int64, device="cpu", pin_memory=True)
    weights = torch.zeros((2, config.top_k), dtype=torch.float32, device="cpu", pin_memory=True)
    weights[:, 0] = 1.0
    count = torch.empty(1, dtype=torch.int32, device="cpu", pin_memory=True)
    task = probe.forward_task(
        count.data_ptr(), ids.size(1), ids.data_ptr(), weights.data_ptr(),
        hidden.data_ptr(), output.data_ptr(), False,
    )
    submit = wrapper.cpu_infer.submit
    sync = wrapper.cpu_infer.sync
    samples = ([], [])
    # Alternate the two cases on the same input/source sequence.
    for i in range(_SAMPLES + 2):
        ids[:, 0].fill_(i % expert_count)
        for case in (i % 2, 1 - i % 2):
            count.fill_(case + 1)
            start = perf_counter_ns()
            submit(task)
            sync(0)
            if i >= 2:
                samples[case].append(perf_counter_ns() - start)
    return tuple(max(1, int(median(values))) for values in samples)


def _measure_copy(plan, transport, stream, device, expert_count):
    # Keep one scratch slot per physical field. The copied bytes never
    # replace resident weights or update the residency map.
    scratch = [
        torch.empty(sizes[0], dtype=torch.uint8, device=device)
        for _, _, sizes, _ in plan.fields
    ]
    batches = [
        [
            H2DCopyTask(sources[eid], target.untyped_storage(), sizes[eid])
            for (sources, _, sizes, _), target in zip(plan.fields, scratch)
        ]
        for eid in range(expert_count)
    ]
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    with torch.npu.stream(stream):
        for i in range(_SAMPLES + 2):
            start.record(stream)
            transport.copy_batch(batches[i % expert_count])
            end.record(stream)
            end.synchronize()
            if i >= 2:
                samples.append(round(start.elapsed_time(end) * 1_000_000))
    return max(1, int(median(samples)))


def calibrate_cpu_moe(backend, plan, transport, stream, device):
    """Measure one expert, duplicate routes to it, and one expert upload."""
    expert_count = min(plan.num_experts, _SOURCE_EXPERTS)
    first, repeated = _measure_cpu(backend, expert_count)
    copy = _measure_copy(plan, transport, stream, device, expert_count)
    return CpuMoeCosts(first, repeated, copy)
