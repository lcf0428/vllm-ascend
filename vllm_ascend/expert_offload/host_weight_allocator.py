"""Place DMA source buffers without moving the CPU expert compute workers."""

import ctypes
import logging
import os
from pathlib import Path

_LOG = logging.getLogger(__name__)

def parse_cpu_list(value: str) -> set[int]:
    cpus = set()
    for span in value.strip().split(","):
        if span:
            first, _, last = span.partition("-")
            cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def eligible_affinity_cpus(candidates: set[int]) -> set[int]:
    """Check explicitly requested CPUs, including CPUs outside inherited affinity."""
    inherited = os.sched_getaffinity(0)
    if candidates <= inherited:
        return candidates
    # Linux intersects the requested mask with the effective cpuset. Keep
    # inherited CPUs in the probe so an empty intersection is still readable.
    os.sched_setaffinity(0, candidates | inherited)
    try:
        return candidates & os.sched_getaffinity(0)
    finally:
        os.sched_setaffinity(0, inherited)


def host_weight_allocation_cpu(node: int) -> int:
    cpus = parse_cpu_list(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text())
    allowed = cpus & os.sched_getaffinity(0)
    if not allowed:
        allowed = eligible_affinity_cpus(cpus)
    if not allowed:
        raise ValueError(f"CPU MoE host_weight_numa_node={node} has no eligible allocation CPU")
    return min(allowed)


class HostWeightAllocator:
    """Temporarily select the allocation CPU and thread memory policy.

    The CPU backend can leave its caller under strict MPOL_BIND. Changing
    affinity alone does not override that policy. Restore both on every exit;
    CPU compute workers keep their existing NUMA configuration.
    None preserves the transport's existing allocation behavior.
    """

    _MASK_BITS = 1024

    def __init__(self, numa_node=None, *, allow_policy_fallback=False):
        self.cpu = None
        self._allow_policy_fallback = allow_policy_fallback
        self._policy_unavailable = False
        if numa_node is not None:
            if isinstance(numa_node, bool) or not isinstance(numa_node, int) or not 0 <= numa_node < self._MASK_BITS:
                raise ValueError("host_weight_numa_node must be a valid non-negative NUMA index or None")
            self.cpu = host_weight_allocation_cpu(numa_node)
            self.node = numa_node
            self._numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
            self._numa.get_mempolicy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
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
        policy = None
        try:
            os.sched_setaffinity(0, {self.cpu})
            if not self._policy_unavailable:
                try:
                    policy = self._save_policy()
                    self._select_policy()
                except PermissionError:
                    if not self._allow_policy_fallback:
                        raise
                    policy = None
                    self._policy_unavailable = True
                    _LOG.warning("CPU MoE host memory policy is blocked; allocating from CPU %d on NUMA node %d",
                                 self.cpu, self.node)
            try:
                return allocate(shape, dtype)
            except Exception as error:
                try:
                    if hasattr(error, "add_note"):
                        mode, mask = self._save_policy()
                        memory = [" ".join(line.strip() for line in Path(f"/sys/devices/system/node/node{n}/meminfo").read_text().splitlines() if "MemFree:" in line) for n in (0, 1)]
                        error.add_note(f"DMA host allocation: shape={shape} dtype={dtype} affinity={sorted(os.sched_getaffinity(0))} mempolicy={mode} nodemask={list(mask)} free={memory}")
                except Exception:
                    pass
                raise
        finally:
            try:
                if policy is not None:
                    self._restore_policy(policy)
            finally:
                os.sched_setaffinity(0, previous)
