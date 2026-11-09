"""Profile-only host DMA timestamps; buffered and flushed outside inference."""

import json
import os
import threading
import time
from pathlib import Path


def stamp(record, name):
    if record is not None:
        record[name + "_ns"] = time.monotonic_ns()
        record[name + "_cpu_ns"] = time.thread_time_ns()


class DmaSubmitTrace:
    def __init__(self):
        self.enabled = False
        self.records = []
        self.inflight = 0
        self.condition = threading.Condition()
        self.directory = None
        self.call_profiles = []
        self.active_call_profile = None
        self.call_profiler_type = None

    def set_capture(self, enabled, directory):
        with self.condition:
            if enabled:
                if not directory:
                    raise ValueError("DMA timing requires the profiler output directory")
                self.directory = Path(directory)
                self.call_profiles = []
                self.call_profiler_type = None
                if (self.directory / "ENABLE_DMA_CPROFILE").exists():
                    import cProfile

                    self.call_profiler_type = cProfile.Profile
                self.records = []
                self.enabled = True
                return
            self.enabled = False
            if not self.condition.wait_for(lambda: self.inflight == 0, timeout=10):
                raise RuntimeError("DMA timing still has an in-flight submit")
            records, self.records = self.records, []
        if self.directory is not None and records:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / f"cpu_dma_submit_{os.getpid()}.jsonl").open("a") as f:
                for r in records:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            if self.call_profiles:
                import io
                import pstats

                stream = io.StringIO()
                stats = pstats.Stats(self.call_profiles[0], stream=stream)
                for profile in self.call_profiles[1:]:
                    stats.add(profile)
                stats.dump_stats(str(self.directory / f"cpu_dma_calls_{os.getpid()}.pstats"))
                stats.sort_stats("cumulative").print_stats(45)
                stats.sort_stats("tottime").print_stats(35)
                (self.directory / f"cpu_dma_calls_{os.getpid()}.txt").write_text(stream.getvalue())

    def begin(self, record):
        with self.condition:
            if not self.enabled:
                return None
            self.inflight += 1
        if self.call_profiler_type is not None and len(self.call_profiles) < 16:
            profile = self.call_profiler_type()
            self.call_profiles.append(profile)
            self.active_call_profile = profile
            record["cprofile_sampled"] = True
            profile.enable()
        return record

    def end(self, record):
        if record is not None:
            if self.active_call_profile is not None:
                self.active_call_profile.disable()
                self.active_call_profile = None
            with self.condition:
                self.records.append(record)
                self.inflight -= 1
                self.condition.notify_all()
