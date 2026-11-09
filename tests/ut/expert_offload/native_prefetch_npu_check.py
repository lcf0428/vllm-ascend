"""Real NPU graph tests: native notification, generation reuse and DMA join."""
import torch
import torch_npu
import _native_dispatch_fixture as fixture
from _vllm_ascend_lrc import NativeCpuDispatcher, NativeHostPlanner, Policy

torch.npu.set_device(0)
stream = torch_npu.npu.SyncLaunchStream(device=0)
engine = NativeCpuDispatcher(fixture.api(), int(stream.npu_stream), int(stream.npu_stream), 62, 63)
owners = []


def buffers():
    dflag = torch.zeros(1, dtype=torch.int32, device='npu')
    hflag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    ids = torch.tensor([[0]], dtype=torch.int32, pin_memory=True)
    weights = torch.ones((1, 1), dtype=torch.float32, pin_memory=True)
    mapping = torch.tensor([0, -1, -1, -1], dtype=torch.int32, pin_memory=True)
    ready = torch_npu.npu.ExternalEvent()
    ready.reset()
    owners.append((dflag, hflag, ids, weights, mapping, ready))
    return owners[-1]


try:
    # Resident prediction: native notification and completion progress under GIL.
    pd, ph, pi, pw, pm, pe = buffers()
    cd, ch, ci, cw, cm, ce = buffers()
    weight_event = torch_npu.npu.ExternalEvent()
    weight_event.reset()
    planner = NativeHostPlanner(4, 1, 0, 0, 1, 0, 'load_balance', None, [0.] * 4)
    task = fixture.task()
    address = fixture.address(task)
    engine.add_prefetch(ph.data_ptr(), pi.data_ptr(), pw.data_ptr(), pm.data_ptr(), int(pe.npu_event),
                        1, 1, 0, 4, 4, 1, planner)
    engine.add(ch.data_ptr(), address, (fixture.entry(), address), ci.data_ptr(), cw.data_ptr(),
               cm.data_ptr(), int(ce.npu_event), int(weight_event.npu_event), 1, 1, 0, 4, 4, planner)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    out = torch.zeros(1, device='npu')
    with torch.npu.graph(graph):
        pe.reset()
        ce.reset()
        weight_event.reset()
        pd.add_(1)
        ph.copy_(pd, non_blocking=True)
        pe.wait()
        cd.add_(1)
        ch.copy_(cd, non_blocking=True)
        ce.wait()
        weight_event.wait()
        out.add_(1)
    for generation in range(1, 6):
        graph.replay()
        assert fixture.wait_holding_gil(task, generation)
        torch.npu.synchronize()
        assert out.item() == generation
    assert engine.prefetch_stats() == (5, 0)
    print('PREFETCH_NO_COPY_GIL_HELD_5_GENERATIONS_OK', flush=True)

    # A miss must not release its consumer until both weights and mapping arrive.
    dd, dh, ids, weights, mapping, done = buffers()
    ids.fill_(2)
    device_mapping = torch.tensor([0, -1, -1, -1], dtype=torch.int32, device='npu')
    source_mapping = device_mapping.clone()
    source_ids = torch.tensor([[2]], dtype=torch.int32, device='npu')
    source_weights = torch.tensor([37.], dtype=torch.float32, pin_memory=True)
    device_weights = torch.zeros(1, device='npu')
    observed = torch.zeros(1, device='npu')
    state = Policy(1, 4, 32, .9, 1., .5, .3, .01, False)
    load_planner = NativeHostPlanner(4, 1, 0, 0, 1, 0, 'load_balance', state, [0.] * 4)
    channel = engine.add_prefetch(dh.data_ptr(), ids.data_ptr(), weights.data_ptr(), mapping.data_ptr(),
                                   int(done.npu_event), 1, 1, 0, 4, 4, 1, load_planner)
    torch.npu.synchronize()
    dma_graph = torch.npu.NPUGraph()
    with torch.npu.graph(dma_graph):
        done.reset()
        device_mapping.copy_(source_mapping)
        device_weights.zero_()
        ids.copy_(source_ids, non_blocking=True)
        mapping.copy_(device_mapping, non_blocking=True)
        dd.add_(1)
        dh.copy_(dd, non_blocking=True)
        done.wait()
        observed.copy_(device_weights + device_mapping[2].float())
    for generation in range(1, 6):
        dma_graph.replay()
        index, loads = engine.next_weight()
        assert index == channel and loads == [(0, 2)]
        with torch.npu.stream(stream):
            device_weights.copy_(source_weights, non_blocking=True)
            device_mapping.copy_(mapping, non_blocking=True)
            done.record(stream)
        torch.npu.synchronize()
        assert observed.item() == 37.
        assert device_mapping.cpu().tolist() == [-1, -1, 0, -1]
        assert state.step(0) == 0
    assert engine.prefetch_stats() == (10, 5)
    print('PREFETCH_WEIGHTS_AND_MAPPING_JOIN_5_GENERATIONS_OK', engine.prefetch_stats(), flush=True)
finally:
    engine.close()
