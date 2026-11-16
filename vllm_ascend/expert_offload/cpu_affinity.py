# SPDX-License-Identifier: Apache-2.0

"""Validate the CPU MoE worker's topology before any weights are loaded."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .host_weight_allocator import eligible_affinity_cpus, host_weight_allocation_cpu, parse_cpu_list
from .npu_topology import discover_npu_numa

_LOG = logging.getLogger(__name__)
_NATIVE_DISPATCH_CPU_LIMIT = 1024  # CPU_SETSIZE in the native dispatcher.


@dataclass(frozen=True)
class CpuMoeAffinity:
    numa_nodes: tuple[int, ...]
    compute_cpus: tuple[tuple[int, ...], ...]
    host_weight_numa_node: int
    poll_cpu: int
    completion_cpu: int
    dma_cpu: int


def _physical_core(cpu: int) -> frozenset[int]:
    siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    return frozenset(parse_cpu_list(siblings.read_text()))


def _cpu_override(name: str) -> int | None:
    value = os.environ.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError as error:
        raise ValueError(f"{name} must be a Linux logical CPU ID, got {value!r}") from error


def _validate_overrides(overrides, cores):
    assigned = {}
    used = set()
    for role, cpu in overrides.items():
        if cpu is None:
            continue
        if not 0 <= cpu < _NATIVE_DISPATCH_CPU_LIMIT:
            raise ValueError(f"CPU MoE {role} CPU {cpu} is outside the native dispatcher CPU mask")
        if cpu not in cores:
            if role != "dma" or cpu not in eligible_affinity_cpus({cpu}):
                raise ValueError(f"CPU MoE {role} CPU {cpu} is outside its allowed CPU set")
            cores[cpu] = _physical_core(cpu)
        core = cores[cpu]
        if core in used:
            raise ValueError(f"CPU MoE {role} CPU {cpu} shares a physical core with another CPU MoE role")
        assigned[role] = cpu
        used.add(core)
    return assigned


def _reserve_service_cores(inherited, cores, inventory, roles, overrides):
    assigned = roles.copy()
    used = {cores[cpu] for cpu in assigned.values()}
    compute_cores = {cores[cpu] for cpu in inventory}
    # First use cores outside this compute pool, then reserve from its tail.
    service_cpus = sorted(
        (cpu for cpu in inherited if cpu < _NATIVE_DISPATCH_CPU_LIMIT),
        key=lambda cpu: (cores[cpu] in compute_cores, -cpu),
    )
    for role in overrides:
        if role in assigned:
            continue
        cpu = next((cpu for cpu in service_cpus if cores[cpu] not in used), None)
        if cpu is None:
            raise ValueError(f"CPU MoE has no free physical core for {role} in the worker CPU affinity")
        assigned[role] = cpu
        used.add(cores[cpu])
    return assigned, tuple(cpu for cpu in inventory if cores[cpu] not in used)


def _reserve_near_npu(inherited, cores, inventory, local_cpus, manual, minimum_compute):
    roles = manual.copy()
    used = {cores[cpu] for cpu in roles.values()}
    compute_cores = {cores[cpu] for cpu in inventory}
    candidates = sorted(
        (cpu for cpu in inherited if cpu < _NATIVE_DISPATCH_CPU_LIMIT),
        key=lambda cpu: (cpu not in local_cpus, cores[cpu] in compute_cores, -cpu),
    )
    # Manual roles are reserved first. DMA gets the first local core among
    # the automatic roles, then poll and completion use what remains.
    for role in ("dma", "poll", "completion"):
        if role in roles:
            continue
        spare_compute = len(compute_cores - used) > minimum_compute
        cpu = next((cpu for cpu in candidates if cores[cpu] not in used
                    and (spare_compute or cores[cpu] not in compute_cores)), None)
        if cpu is None:
            return None
        roles[role] = cpu
        used.add(cores[cpu])
    compute = tuple(cpu for cpu in inventory if cores[cpu] not in used)
    return (roles, compute) if len(compute) >= minimum_compute else None


def plan_cpu_moe_affinity(config, *, resolve_physical_npu=None) -> CpuMoeAffinity:
    from kt_kernel import kt_kernel_ext

    inventory_for_nodes = getattr(kt_kernel_ext, "available_worker_pool_cpus", None)
    if inventory_for_nodes is None:
        raise RuntimeError("Rebuild kt-kernel with CPU MoE automatic affinity support")
    if config.threadpool_count != 1:
        raise ValueError("Shared NZ CPU requires one NUMA pool")

    allowed_memory = parse_cpu_list(next(
        line.split(":", 1)[1]
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("Mems_allowed_list:")
    ))
    weight_node = config.host_weight_numa_node
    required_nodes = set(config.numa_nodes or ())
    if weight_node is not None:
        required_nodes.add(weight_node)
    if missing := required_nodes - allowed_memory:
        raise ValueError(f"CPU MoE NUMA nodes {sorted(missing)} are outside the allowed memory nodes")
    if weight_node is not None:
        host_weight_allocation_cpu(weight_node)

    inherited = os.sched_getaffinity(0)
    cores = {cpu: _physical_core(cpu) for cpu in inherited}
    overrides = {
        "poll": _cpu_override("KT_MOE_POLL_CPU"),
        "completion": _cpu_override("KT_MOE_COMPLETION_CPU"),
        "dma": config.native_dispatch_dma_cpu,
    }
    manual = _validate_overrides(overrides, cores)
    nodes = list(config.numa_nodes or sorted(allowed_memory))
    inventories = inventory_for_nodes(nodes)
    if not any(inventories):
        raise ValueError("CPU MoE has no eligible compute CPUs on the allowed NUMA nodes")

    requested = config.cpuinfer_threads
    near = None
    if resolve_physical_npu is not None:
        topology = discover_npu_numa(resolve_physical_npu)
        if topology is None:
            _LOG.info("CPU MoE NPU topology unavailable or ambiguous; selecting NUMA by available cores")
        else:
            node, local_cpus = topology
            inventory = inventories[nodes.index(node)] if node in nodes else ()
            if inventory:
                near = _reserve_near_npu(inherited, cores, inventory, local_cpus, manual, requested or 1)
            if near is None:
                _LOG.info("CPU MoE NPU-affine NUMA %d cannot satisfy the configured affinity; "
                          "selecting NUMA by available cores", node)
    if near is not None:
        roles, compute = near
        _LOG.info("CPU MoE selected NPU-affine NUMA %d", node)
    else:
        options = []
        for node, inventory in zip(nodes, inventories):
            if inventory:
                roles, compute = _reserve_service_cores(inherited, cores, inventory, manual, overrides)
                options.append((node, roles, compute))
        node, roles, compute = max(options, key=lambda option: (len(option[2]), -option[0]))
    if not compute or (requested is not None and requested > len(compute)):
        demand = "at least one" if requested is None else str(requested)
        raise ValueError(
            f"CPU MoE needs {demand} compute cores; NUMA node {node} has {len(compute)} "
            "available after reserving poll, completion and DMA cores"
        )
    return CpuMoeAffinity(
        (node,), (compute[:requested],), node if weight_node is None else weight_node,
        roles["poll"], roles["completion"], roles["dma"],
    )
