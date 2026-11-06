"""Real NPU streams/replays with a small deterministic CPU expert stand-in.

Requires the normal poll library, kt-kernel notifier runtime, C++ LRC, and
mapped-combine extension. No model weights or callback dispatch are used.
"""

import os
import time
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("slow", ["cpu", "weights"])
def test_host_dma_and_cpu_output_across_graph_replays(tmp_path, monkeypatch, slow):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("kt_kernel")
    pytest.importorskip("_vllm_ascend_lrc")
    if not torch.npu.is_available() or not os.environ.get("KT_MOE_POLL_LIBRARY"):
        pytest.skip("Requires NPU and KT_MOE_POLL_LIBRARY")
    from vllm_ascend.expert_offload.cpu_backend.kt_kernel_backend import KtKernelBackendConfig, KtKernelCPUExpertBackend
    from vllm_ascend.expert_offload.host_transfer import HostExpertTransfer, HostWeightWorker
    from vllm_ascend.expert_offload.hybrid_executor import HybridExpertExecutor
    from vllm_ascend.expert_offload.lrc_policy import CppLRCExpertCachePolicy

    for key in ("KT_MOE_LIFECYCLE_TRACE", "VLLM_ASCEND_CPU_MOE_TRACE"):
        monkeypatch.delenv(key, raising=False)
    torch.npu.set_device(0)
    weight_path = tmp_path / "standin.gguf"
    weight_path.touch()

    class Wrapper:
        def __init__(self, **_):
            self.hidden = torch.zeros(1, 64, dtype=torch.bfloat16, pin_memory=True)
            self.ids = torch.zeros(1, 4, dtype=torch.int64, pin_memory=True)
            self.weights = torch.zeros(1, 4, pin_memory=True)
            self.output = torch.zeros_like(self.hidden, pin_memory=True)
            self.device_output = torch.zeros(1, 64, dtype=torch.bfloat16, device="npu")

        def load_weights(self, *_):
            pass

        def set_capture_batch_sizes(self, *_):
            pass

        def submit_forward(self, *_):
            raise AssertionError("Host path must submit pinned inputs")

        def sync_forward(self, *_):
            raise AssertionError("Unexpected legacy sync")

        def copy_inputs_to_cpu_buffers(self, hidden, ids, weights):
            self.hidden.copy_(hidden, non_blocking=True)
            self.ids.copy_(ids, non_blocking=True)
            self.weights.copy_(weights, non_blocking=True)

        def forward_on_pinned_buffers(self, *_, **__):
            pass

        def drain_pinned_forward(self):
            if slow == "cpu":
                time.sleep(0.005)
            coefficient = torch.where(self.ids >= 0, (self.ids + 1) * self.weights, 0).sum()
            self.output.copy_(self.hidden * coefficient)

        def get_forward_output_buffers(self, _):
            return self.output, self.device_output

        def copy_forward_output_to_device(self, _):
            raise AssertionError("CPU output must use mapped combine")

    with torch.inference_mode():
        backend = KtKernelCPUExpertBackend(
            KtKernelBackendConfig(0, 6, 4, 64, 8, str(weight_path)), wrapper_factory=Wrapper
        )
        monkeypatch.setattr(backend, "_pin_graph_batch_size", lambda _: None)
        monkeypatch.setattr(
            backend, "_pinned_route_buffers", lambda _: (backend._wrapper.ids, backend._wrapper.weights)
        )
        layer = SimpleNamespace(log2phy=torch.tensor([0, 1, -1, -1, -1, -1], dtype=torch.int32, device="npu"))
        slots = torch.tensor([1.0, 2.0], device="npu")
        sources = torch.arange(1, 7, dtype=torch.float32, pin_memory=True)

        def load(_layer, _idx, loads):
            if slow == "weights":
                time.sleep(0.005)
            for slot, expert in loads:
                slots[slot : slot + 1].copy_(sources[expert : expert + 1], non_blocking=True)

        manager = SimpleNamespace(
            moe_layers=[layer],
            num_total_experts=6,
            num_device_experts_for_layer=lambda _: 2,
            offload_config=SimpleNamespace(
                cpu_moe=SimpleNamespace(
                    transfer_policy="first_miss",
                    balance_cpu_expert_us=70.0,
                    balance_copy_us=450.0,
                    balance_max_transfers=32,
                )
            ),
            cache_policy=CppLRCExpertCachePolicy(num_layers=1, num_experts=6, cache_size=2, topk=4),
            _load_expert_weights_into_slots=load,
            _synchronize_h2d=lambda stream: stream.synchronize(),
        )
        worker = HostWeightWorker(manager)
        backend._host_transfer = HostExpertTransfer(manager, layer, backend, worker)
        executor = HybridExpertExecutor(backend, 0)
        hidden = torch.ones(1, 64, dtype=torch.bfloat16, device="npu")
        ids = torch.tensor([[0, 3, 4, 5]], dtype=torch.int64, device="npu")
        weights = torch.full((1, 4), 0.25, device="npu")

        def forward():
            plan = executor.prepare(hidden_states=hidden, topk_ids=ids, topk_weights=weights, log2phy=layer.log2phy)
            physical = layer.log2phy[plan.npu_topk_ids.long()].clamp(min=0).long()
            coefficient = (slots[physical] * plan.npu_topk_weights).sum()
            return executor.finish(hidden * coefficient, plan.cpu_task)

        try:
            forward()  # eager uses the same planner and DMA protocol
            torch.npu.synchronize()
            graphs, outputs = [], []
            for _ in range(2):
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = forward()
                graphs.append(graph)
                outputs.append(output)
            for step in range(30):
                routes = [[(step + i) % 6 for i in range(4)]]
                ids.copy_(torch.tensor(routes, dtype=torch.int64, device="npu"))
                hidden.fill_(1 + step % 3)
                backend._wrapper.output.fill_(-777)
                index = step % 2
                graphs[index].replay()
                torch.npu.synchronize()
                expected = (1 + step % 3) * sum(e + 1 for e in routes[0]) / 4
                torch.testing.assert_close(
                    outputs[index].cpu(), torch.full((1, 64), expected, dtype=torch.bfloat16), rtol=0, atol=0
                )
        finally:
            worker.close()
