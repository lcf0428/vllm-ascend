"""Host planning/DMA with a weight event independent of CPU output readiness."""

from __future__ import annotations

import atexit
import logging
import queue
import threading
from dataclasses import dataclass

import torch
import torch_npu

from .host_transfer_policy import HostTransferPolicy

_LOG = logging.getLogger(__name__)


@dataclass
class WeightRequest:
    transfer: object
    mapping_host: object
    mapping_device: object
    loads: tuple
    ready: object


class HostWeightWorker:
    """One uncaptured SyncLaunchStream; never waits for CPU expert compute."""

    def __init__(self, manager):
        self.manager = manager
        self.device = torch.npu.current_device()
        self.stream = torch_npu.npu.SyncLaunchStream(device=self.device)
        self.jobs = queue.SimpleQueue()
        self._close_lock = threading.Lock()
        self._closed = False
        self._error = None
        self.thread = threading.Thread(target=self._run, name="cpu_expert_dma", daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("Expert DMA failed; weight readiness was not published") from self._error
        if self._closed:
            raise RuntimeError("Expert DMA worker is closed")

    def submit(self, request):
        self.raise_if_failed()
        self.jobs.put(request)

    def _process(self, request):
        transfer = request.transfer
        with torch.npu.stream(self.stream):
            self.manager._load_expert_weights_into_slots(transfer.layer, transfer.layer_idx, request.loads)
            request.mapping_device.copy_(request.mapping_host, non_blocking=True)
            # Record on DMA itself after weights, quant attributes and mapping.
            # Never use the independent CPU-output completion stream here.
            request.ready.record(self.stream)

    @torch.inference_mode()
    def _run(self):
        try:
            torch.npu.set_device(self.device)
            while True:
                request = self.jobs.get()
                if request is None:
                    return
                self._process(request)
                del request
        except BaseException as exc:
            self._error = exc
            # Keep failed events unsignalled: consumers must not read stale
            # weights. New host submissions raise; an in-flight graph follows
            # the runtime's configured task timeout/error handling.
            _LOG.exception("Expert DMA failed; weight readiness was not published")

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self.raise_if_failed()
            # Poll/CPU/DMA consumers remain alive until live replays retire.
            torch.npu.synchronize(self.device)
            self.jobs.put(None)
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("Expert DMA worker still owns live requests")
            self.manager._synchronize_h2d(self.stream)
            self._closed = True
            atexit.unregister(self.close)


class HostTransferContext:
    """Capture-site owners, kept alive for all replays of their graph."""

    def __init__(self, transfer, hidden):
        self.transfer, self.hidden = transfer, hidden
        self.mapping = torch.empty(
            transfer.layer.log2phy.shape, dtype=transfer.layer.log2phy.dtype, device="cpu", pin_memory=True
        )
        self.ready = torch_npu.npu.ExternalEvent()
        if not self.ready.npu_event:
            raise RuntimeError("Host DMA requires value-based ExternalEvent support")

    def stage(self, stream):
        # Captured reset precedes sequence publication on EVERY replay.
        self.ready.reset(stream)
        self.mapping.copy_(self.transfer.layer.log2phy, non_blocking=True)

    def prepare(self):
        transfer = self.transfer
        transfer.worker.raise_if_failed()
        ids, weights = transfer.backend._pinned_route_buffers(self.hidden)
        plan = transfer.policy.plan(ids.tolist(), weights.float().tolist(), self.mapping.tolist())
        # Full routes are observed first; CPU then computes only unassigned
        # misses. Original device routes remain untouched.
        ids.copy_(torch.tensor(plan.cpu_ids, dtype=ids.dtype).reshape(ids.shape))
        weights.copy_(torch.tensor(plan.cpu_weights, dtype=weights.dtype).reshape(weights.shape))
        self.mapping.copy_(torch.tensor(plan.mapping, dtype=self.mapping.dtype))
        transfer.worker.submit(WeightRequest(transfer, self.mapping, transfer.layer.log2phy, plan.loads, self.ready))


class HostExpertTransfer:
    def __init__(self, manager, layer, backend, worker):
        self.layer, self.backend, self.worker = layer, backend, worker
        self.layer_idx = manager.moe_layers.index(layer)
        self._eager_contexts = {}
        self.policy = HostTransferPolicy(
            experts=manager.num_total_experts,
            slots=manager.num_device_experts_for_layer(self.layer_idx),
            config=manager.offload_config.cpu_moe,
            lrc=manager.cache_policy,
            layer_idx=self.layer_idx,
        )

    def context(self, hidden, *, eager=False):
        if not eager:
            return HostTransferContext(self, hidden)
        # Retain pinned mapping/event owners beyond asynchronous eager H2D.
        # The next use first synchronizes its input stream after prior work.
        key = (tuple(hidden.shape), hidden.dtype)
        context = self._eager_contexts.get(key)
        if context is None:
            context = HostTransferContext(self, hidden)
            self._eager_contexts[key] = context
        context.hidden = hidden
        return context


def attach_host_transfers(manager):
    """Called after weights/hot preload are final, before warmup/capture."""
    cpu = manager.offload_config.cpu_moe
    if not cpu.enabled or cpu.transfer_scheduler != "cpu":
        return
    if manager.ep_size != 1 or manager.enable_multi_card:
        raise ValueError("Host hybrid DMA currently requires single-card execution")
    from .h2d_transfer import TorchCopyH2DTransport
    from .native_mxfp4_h2d import NativeMXFP4H2DTransport

    transport = manager._get_h2d_transport()
    base = transport.base if isinstance(transport, NativeMXFP4H2DTransport) else transport
    if not isinstance(base, TorchCopyH2DTransport):
        raise ValueError("Host hybrid DMA requires h2d_backend=torch")
    manager._host_weight_worker = HostWeightWorker(manager)
    for layer in manager.moe_layers:
        executor = getattr(layer, "_hybrid_expert_executor", None)
        if executor is not None:
            executor.backend._host_transfer = HostExpertTransfer(
                manager, layer, executor.backend, manager._host_weight_worker
            )
