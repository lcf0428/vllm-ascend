"""Call once after profiler start and once BEFORE profiler stop, in worker PID.

Import via an absolute file path or copy into the runtime package. Never call
outside an active torch profiler. Both markers must survive in exported trace.
The Ascend exporter must already map hardware timestamps onto host time.
"""
import json
import os
import time
import uuid
from pathlib import Path


def record_clock_anchor(tag, directory):
    import torch
    # Warm frontend before measuring: first record_function can be expensive.
    with torch.profiler.record_function("CPU_MOE_CLOCK_WARMUP"):
        pass
    name = f"CPU_MOE_CLOCK_ANCHOR_{os.getpid()}_{uuid.uuid4().hex}"
    before = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    with torch.profiler.record_function(name):
        after_enter = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    record = dict(marker_name=name, pid=os.getpid(), tag=tag,
                  before_raw_ns=before, after_enter_raw_ns=after_enter)
    path = Path(directory) / f"cpu_clock_anchors_{os.getpid()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(record) + "\n")
    return record
