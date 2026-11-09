"""Place DMA source buffers without moving the CPU expert compute workers."""

import ctypes
import os
from pathlib import Path


class HostWeightAllocator:
    """Temporarily select the allocation CPU and thread memory policy.

    The CPU backend can leave its caller under strict MPOL_BIND. Changing
    affinity alone does not override that policy. Restore both on every exit;
    GGUF loading and compute workers keep their existing NUMA configuration.
    None preserves the transport's existing allocation behavior.
    """

    _MASK_BITS = 1024

    def __init__(self, numa_node=None):
        self.cpu = None
        if numa_node is not None:
            if isinstance(numa_node, bool) or not isinstance(numa_node, int) or not 0 <= numa_node < self._MASK_BITS:
                raise ValueError("host_weight_numa_node must be a valid non-negative NUMA index or None")
            cpus = Path(f"/sys/devices/system/node/node{numa_node}/cpulist").read_text().strip()
            if not cpus:
                raise ValueError(f"NUMA node {numa_node} has no allocation CPU")
            self.cpu = int(cpus.split(",")[0].split("-")[0])
            self.node = numa_node
            self._numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
            self._numa.get_mempolicy.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.c_ulong,
                ctypes.c_void_p,
                ctypes.c_ulong,
            ]
            self._numa.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]

    def _check(self, result):
        if result:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    def _save_policy(self):
        word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        mask = (ctypes.c_ulong * (self._MASK_BITS // word_bits))()
        mode = ctypes.c_int()
        self._check(self._numa.get_mempolicy(ctypes.byref(mode), mask, self._MASK_BITS, None, 0))
        return mode.value, mask

    def _select_policy(self):
        word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        mask = (ctypes.c_ulong * (self._MASK_BITS // word_bits))()
        mask[self.node // word_bits] = 1 << (self.node % word_bits)
        # MPOL_PREFERRED: prefer this node without prohibiting OS fallback.
        self._check(self._numa.set_mempolicy(1, mask, self._MASK_BITS))

    def _restore_policy(self, saved):
        mode, mask = saved
        # MPOL_DEFAULT requires a null nodemask.
        self._check(self._numa.set_mempolicy(mode, None if mode == 0 else mask, self._MASK_BITS))

    def allocate(self, allocate, shape, dtype):
        if self.cpu is None:
            return allocate(shape, dtype)
        previous = os.sched_getaffinity(0)
        policy = self._save_policy()
        try:
            os.sched_setaffinity(0, {self.cpu})
            self._select_policy()
            try:
                return allocate(shape, dtype)
            except Exception as error:
                try:
                    if hasattr(error, "add_note"):
                        mode, mask = self._save_policy()
                        memory = [
                            " ".join(
                                line.strip()
                                for line in Path(f"/sys/devices/system/node/node{n}/meminfo").read_text().splitlines()
                                if "MemFree:" in line
                            )
                            for n in (0, 1)
                        ]
                        error.add_note(
                            f"DMA host allocation: shape={shape} dtype={dtype} "
                            f"affinity={sorted(os.sched_getaffinity(0))} mempolicy={mode} "
                            f"nodemask={list(mask)} free={memory}"
                        )
                except Exception:
                    pass
                raise
        finally:
            try:
                self._restore_policy(policy)
            finally:
                os.sched_setaffinity(0, previous)
