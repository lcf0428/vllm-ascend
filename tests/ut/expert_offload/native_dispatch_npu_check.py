"""Small real graph check; run only on the task's selected idle NPU/context."""

import _native_dispatch_fixture as fixture
import torch
import torch_npu
from _vllm_ascend_lrc import NativeCpuDispatcher, NativeHostPlanner

print("DEVICE", flush=True)
torch.npu.set_device(0)
device_flag = torch.zeros(1, dtype=torch.int32, device="npu")
host_flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
ids = torch.tensor([[2]], dtype=torch.int64, pin_memory=True)
weights = torch.ones((1, 1), dtype=torch.float32, pin_memory=True)
mapping = torch.tensor([0, -1, -1, -1], dtype=torch.int32, pin_memory=True)
output = torch.zeros(1, device="npu")
event = torch_npu.npu.ExternalEvent()
weight_ready = torch_npu.npu.ExternalEvent()
stream = torch_npu.npu.SyncLaunchStream(device=0)
print("EVENT_INIT", flush=True)
event.reset()
weight_ready.reset()
torch.npu.synchronize()
planner = NativeHostPlanner(4, 1, 0, 0, 1, 0, "load_balance", None, [0.0] * 4)
task = fixture.task()
address = fixture.address(task)
print("ENGINE_INIT", flush=True)
engine = NativeCpuDispatcher(fixture.api(), int(stream.npu_stream), int(stream.npu_stream), 62, 63)
print("REGISTER", flush=True)
engine.add(
    host_flag.data_ptr(),
    address,
    (fixture.entry(), address),
    ids.data_ptr(),
    weights.data_ptr(),
    mapping.data_ptr(),
    int(event.npu_event),
    int(weight_ready.npu_event),
    1,
    1,
    0,
    8,
    4,
    planner,
)
print("GRAPH_INIT", flush=True)
graph = torch.npu.NPUGraph()
print("CAPTURE", flush=True)
with torch.npu.graph(graph):
    event.reset()
    weight_ready.reset()
    device_flag.add_(1)
    host_flag.copy_(device_flag, non_blocking=True)
    event.wait()
    weight_ready.wait()
    output.add_(1)
try:
    for generation in range(1, 6):
        print("REPLAY", generation, flush=True)
        graph.replay()
        assert fixture.wait_holding_gil(task, generation), "Native task could not run while GIL was held"
        torch.npu.synchronize()
        assert output.item() == generation
        assert engine.stats() == (generation, generation)
    print("NATIVE_NPU_GRAPH_GIL_HELD_EXACTLY_ONCE_OK", engine.stats(), flush=True)
finally:
    engine.close()
