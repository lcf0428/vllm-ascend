"""Hardware regression for fixed-slot swaps inside ACL graph host callbacks."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.expert_offload.exclusive_pool import ExclusiveExpertPool
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def test_mxfp4_finalization_preserves_real_nz_slots(monkeypatch):
    """Check the installed torch_npu format representation without a checkpoint."""
    torch.npu.set_device(0)
    weights = []
    for _ in range(2):
        packed = torch.zeros((2, 128, 128), dtype=torch.uint8, device="npu")
        nz = torch_npu.npu_format_cast(
            packed, ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2,
        ).transpose(1, 2)
        weights.append(nz)
    torch.npu.synchronize()
    formats = [torch_npu.get_npu_format(w) for w in weights]
    print(f"Resident MXFP4 formats: {formats!r}")
    before = [(w.data_ptr(), w.stride(), w.untyped_storage().nbytes()) for w in weights]
    layer = SimpleNamespace(w13_weight=weights[0], w2_weight=weights[1])
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.moe_layers = [layer]
    manager.offload_config = SimpleNamespace(exclusive_expert_storage=True)

    def unexpected_cast(*args, **kwargs):
        raise AssertionError(f"Already-NZ resident slots must not be recast: {formats!r}")

    monkeypatch.setattr(torch_npu, "npu_format_cast", unexpected_cast)
    for _ in range(2):
        manager._cast_device_slots_to_mxfp4_nz()
    assert layer.w13_weight is weights[0]
    assert layer.w2_weight is weights[1]
    assert before == [(w.data_ptr(), w.stride(), w.untyped_storage().nbytes()) for w in weights]
    assert formats == [torch_npu.get_npu_format(w) for w in weights]


@pytest.mark.parametrize("capture", [False, True])
def test_exclusive_swap_repeated_routes(capture):
    torch.npu.set_device(0)
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = SimpleNamespace(shard_per_rank=False)
    manager.load_stream = torch_npu.npu.Stream()
    manager.h2d_transport = TorchCopyH2DTransport()
    manager._exclusive_pool = ExclusiveExpertPool()
    fields = ("w13", "w2", "w13_weight_scale", "w2_weight_scale")
    bundles = [None, None] + [{name: torch.full((32, 32), eid + j, dtype=torch.uint8, pin_memory=True)
                              for j, name in enumerate(fields)} for eid in range(2, 6)]
    def initial(value):
        return torch.stack([torch.full((32, 32), value + slot, dtype=torch.uint8, device="npu")
                            for slot in range(2)])
    layer = SimpleNamespace(
        w13_weight=initial(0), w2_weight=initial(1),
        w13_weight_scale=initial(2), w2_weight_scale=initial(3),
        w13_weight_scale_fp32=initial(2).float())
    manager.w13_expert_size_bytes = manager.w2_expert_size_bytes = 1024
    manager.scale_cpu_buffers = {name: [[None] * 6] for name in fields[2:]}
    manager.offset_cpu_buffers = {}
    manager.scale_bias_cpu_buffers = {}
    manager._exclusive_pool.add_layer(
        0, bundles, [0, 1, -1, -1, -1, -1],
        allocate_spare=lambda template: {name: torch.empty_like(t, pin_memory=True)
                                         for name, t in template.items()})
    del bundles
    route = torch.tensor([2, 3], dtype=torch.int32, device="npu")
    route_h = torch.empty(2, dtype=torch.int32, pin_memory=True)
    observed = [torch.empty_like(getattr(layer, name if "scale" in name else name + "_weight"))
                for name in fields]
    weights = [getattr(layer, name if "scale" in name else name + "_weight") for name in fields]
    addresses = [value.data_ptr() for value in weights]
    errors = []

    def exchange(_):
        try:
            loads = [(slot, int(eid)) for slot, eid in enumerate(route_h.tolist())
                     if manager._exclusive_pool.resident[0, slot] != int(eid)]
            manager._load_expert_weights_into_slots(layer, 0, loads)
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
    for ids in [(2, 3), (2, 3), (0, 1), (4, 5), (2, 3), (4, 3), (0, 5)] * 3:
        route.copy_(torch.tensor(ids, dtype=torch.int32, device="npu"))
        if capture:
            graph.replay()
        else:
            record()
        torch.npu.synchronize()
        assert not errors, errors
        for j, output in enumerate(observed):
            expected = torch.stack([torch.full((32, 32), eid + j, dtype=torch.uint8) for eid in ids])
            torch.testing.assert_close(output.cpu(), expected)
        for (_, eid), host in manager._exclusive_pool.host.items():
            for j, name in enumerate(fields):
                torch.testing.assert_close(host[name], torch.full_like(host[name], eid + j))
        torch.testing.assert_close(layer.w13_weight_scale_fp32.cpu(),
                                   torch.stack([torch.full((32, 32), float(eid + 2)) for eid in ids]))
        assert len(manager._exclusive_pool.host) == 4
        assert len(manager._exclusive_pool.spares) == 1
        assert addresses == [value.data_ptr() for value in weights]
