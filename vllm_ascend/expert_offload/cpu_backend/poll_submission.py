
"""Submit graph CPU MoE from an ordered D2H sequence, without ACL callbacks.

The int32 sequence is copied after all input copies on the same stream.
Pinned slots remain alive for the engine lifetime. A missed sequence aborts
rather than using stale inputs.
"""
import threading

import torch

from .native_submission import NativePollWorker

_WORKER = None
_LOCK = threading.Lock()
_GRAPH_SLOTS = 64


def prepare(backend):
    global _WORKER
    device = torch.npu.current_device()
    shape = (_GRAPH_SLOTS, 16)
    backend._poll_sequence_device = torch.zeros(shape, device=torch.device("npu", device), dtype=torch.int32)
    backend._poll_sequence_host = torch.zeros(shape, device="cpu", dtype=torch.int32, pin_memory=True)
    backend._poll_sequence_slots = [
        (backend._poll_sequence_device[i, :1], backend._poll_sequence_host[i, :1])
        for i in range(_GRAPH_SLOTS)
    ]
    backend._poll_sequence_used = 0
    with _LOCK:
        if _WORKER is None:
            _WORKER = NativePollWorker(device, backend.config.poll_cpu, backend.config.completion_cpu)


def record_submission(backend, hidden, stream, ready, before_submit=None):
    slot = backend._poll_sequence_used
    if slot >= _GRAPH_SLOTS:
        raise RuntimeError("CPU input poll graph slot limit reached")
    backend._poll_sequence_used += 1
    device_flag, host_flag = backend._poll_sequence_slots[slot]
    result = _WORKER.register(backend, hidden, stream, ready, host_flag, before_submit)
    device_flag.add_(1)
    host_flag.copy_(device_flag, non_blocking=True)
    return result
