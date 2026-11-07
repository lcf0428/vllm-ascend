"""Hardware regression for fixed-slot swaps inside ACL graph host callbacks."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.expert_offload.exclusive_pool import ExclusiveExpertPool
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport


@pytest.mark.parametrize("capture", [False, True])
def test_exclusive_swap_repeated_routes(capture):
    torch.npu.set_device(0)
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = SimpleNamespace(shard_per_rank=False)
    manager.load_stream = torch_npu.npu.Stream()
    manager.h2d_transport = TorchCopyH2DTransport()
    manager._exclusive_pool = ExclusiveExpertPool()
    fields = ("w13", "w2", "w13_weight_scale", "w2_weight_scale")
    bundles = [None] + [{name: torch.full((32, 32), eid + j, dtype=torch.uint8, pin_memory=True)
                        for j, name in enumerate(fields)} for eid in range(1, 4)]
    layer = SimpleNamespace(
        w13_weight=torch.full((1, 32, 32), 0, dtype=torch.uint8, device="npu"),
        w2_weight=torch.full((1, 32, 32), 1, dtype=torch.uint8, device="npu"),
        w13_weight_scale=torch.full((1, 32, 32), 2, dtype=torch.uint8, device="npu"),
        w2_weight_scale=torch.full((1, 32, 32), 3, dtype=torch.uint8, device="npu"))
    manager.w13_expert_size_bytes = manager.w2_expert_size_bytes = 1024
    manager.scale_cpu_buffers = {name: [[None] * 4] for name in fields[2:]}
    manager.offset_cpu_buffers = {}
    manager.scale_bias_cpu_buffers = {}
    manager._exclusive_pool.add_layer(
        0, bundles, [0, -1, -1, -1],
        allocate_spare=lambda template: {name: torch.empty_like(t, pin_memory=True)
                                         for name, t in template.items()})
    del bundles
    route = torch.tensor([1], dtype=torch.int32, device="npu")
    route_h = torch.empty(1, dtype=torch.int32, pin_memory=True)
    observed = [torch.empty_like(getattr(layer, name if "scale" in name else name + "_weight"))
                for name in fields]
    weights = [getattr(layer, name if "scale" in name else name + "_weight") for name in fields]
    addresses = [value.data_ptr() for value in weights]
    errors = []

    def exchange(_):
        try:
            eid = int(route_h[0])
            if manager._exclusive_pool.resident[0, 0] != eid:
                with torch_npu.npu.stream(manager.load_stream):
                    manager._exclusive_swap(layer, 0, 0, eid)
        except BaseException as error:
            errors.append(error)

    stream = torch_npu.npu.Stream()
    torch_npu.npu._subscribe_report(stream)

    def record():
        route_h.copy_(route, non_blocking=capture)
        if capture:
            torch_npu.npu._launch_host_func(stream, exchange, None)
        else:
            exchange(None)
        for name, output in zip(fields, observed):
            output.copy_(getattr(layer, name if "scale" in name else name + "_weight"))

    torch.npu.synchronize()
    graph = torch.npu.NPUGraph() if capture else None
    if capture:
        with torch.npu.graph(graph, stream=stream):
            record()
    for eid in [1, 2, 0, 3, 2, 1, 0] * 3:
        route.fill_(eid)
        if capture:
            graph.replay()
        else:
            record()
        torch.npu.synchronize()
        assert not errors, errors
        for j, output in enumerate(observed):
            torch.testing.assert_close(output.cpu(), torch.full_like(output, eid + j, device="cpu"))
        assert len(manager._exclusive_pool.host) == 3
        assert len(manager._exclusive_pool.spares) == 1
        assert addresses == [value.data_ptr() for value in weights]
