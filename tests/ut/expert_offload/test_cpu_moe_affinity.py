import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.expert_offload import cpu_affinity


@pytest.fixture
def topology(monkeypatch):
    available = {20, 21, 22, 23, 24, 25, 30, 31, 32}
    nodes = {2: [20, 21, 23, 24, 25], 4: [30, 31, 32]}
    inventory = Mock(side_effect=lambda ids: [[cpu for cpu in nodes[node] if cpu in available] for node in ids])
    module = SimpleNamespace(kt_kernel_ext=SimpleNamespace(available_worker_pool_cpus=inventory))
    monkeypatch.setitem(sys.modules, "kt_kernel", module)
    monkeypatch.setattr(cpu_affinity.os, "sched_getaffinity", lambda _: available.copy(), raising=False)
    for name in ("KT_MOE_POLL_CPU", "KT_MOE_COMPLETION_CPU"):
        monkeypatch.delenv(name, raising=False)

    def read_text(path):
        if path.name == "status":
            return "Mems_allowed_list:\t2,4\n"
        cpu = int(path.parent.parent.name.removeprefix("cpu"))
        return "20,22" if cpu in {20, 22} else str(cpu)

    monkeypatch.setattr(cpu_affinity.Path, "read_text", read_text)
    monkeypatch.setattr(cpu_affinity, "host_weight_allocation_cpu", Mock(return_value=30))
    settings = SimpleNamespace(
        cpuinfer_threads=None,
        threadpool_count=1,
        numa_nodes=None,
        host_weight_numa_node=None,
        native_dispatch_dma_cpu=None,
    )
    return settings, available, inventory


def test_auto_chooses_largest_compute_node_and_reserves_service_cores(topology):
    settings, _, inventory = topology
    plan = cpu_affinity.plan_cpu_moe_affinity(settings)

    inventory.assert_called_once_with([2, 4])
    assert plan.numa_nodes == (2,)
    assert plan.compute_cpus == ((20, 21, 23, 24, 25),)
    assert plan.host_weight_numa_node == 2
    assert (plan.poll_cpu, plan.completion_cpu, plan.dma_cpu) == (32, 31, 30)


def test_single_node_reserves_service_cores_before_counting_compute(topology):
    settings, available, _ = topology
    available.difference_update({30, 31, 32})
    plan = cpu_affinity.plan_cpu_moe_affinity(settings)
    assert plan.compute_cpus == ((20, 21),)
    assert (plan.poll_cpu, plan.completion_cpu, plan.dma_cpu) == (25, 24, 23)


@pytest.fixture
def npu_locality(monkeypatch):
    query = Mock(return_value=(4, {30, 31, 32}))
    monkeypatch.setattr(cpu_affinity, "discover_npu_numa", query)
    return query


def test_prefers_npu_numa_and_reserves_a_local_dma_core(topology, npu_locality):
    def resolve():
        return 7

    plan = cpu_affinity.plan_cpu_moe_affinity(topology[0], resolve_physical_npu=resolve)

    npu_locality.assert_called_once_with(resolve)
    assert plan.numa_nodes == (4,)
    assert plan.host_weight_numa_node == 4
    assert plan.compute_cpus == ((30,),)
    assert (plan.dma_cpu, plan.poll_cpu, plan.completion_cpu) == (32, 31, 25)


def test_preserves_requested_compute_and_moves_poll_and_completion_off_node(topology, npu_locality):
    settings, _, _ = topology
    settings.cpuinfer_threads = 2
    plan = cpu_affinity.plan_cpu_moe_affinity(settings, resolve_physical_npu=lambda: 7)
    assert plan.compute_cpus == ((30, 31),)
    assert (plan.dma_cpu, plan.poll_cpu, plan.completion_cpu) == (32, 25, 24)


@pytest.mark.parametrize("locality", [None, (9, {70, 71})])
def test_missing_or_disallowed_topology_uses_existing_capacity_policy(topology, npu_locality, locality):
    settings = topology[0]
    npu_locality.return_value = locality
    assert cpu_affinity.plan_cpu_moe_affinity(
        settings, resolve_physical_npu=lambda: 7,
    ) == cpu_affinity.plan_cpu_moe_affinity(settings)


def test_affine_node_without_enough_compute_falls_back(topology, npu_locality):
    settings = topology[0]
    settings.cpuinfer_threads = 4
    plan = cpu_affinity.plan_cpu_moe_affinity(settings, resolve_physical_npu=lambda: 7)
    assert plan.numa_nodes == (2,)
    assert plan.compute_cpus == ((20, 21, 23, 24),)


def test_explicit_overrides_win_and_exclude_smt_siblings(topology, monkeypatch):
    settings, _, _ = topology
    settings.cpuinfer_threads = 2
    settings.numa_nodes = (2,)
    settings.host_weight_numa_node = 4
    settings.native_dispatch_dma_cpu = 24
    monkeypatch.setenv("KT_MOE_POLL_CPU", "22")
    monkeypatch.setenv("KT_MOE_COMPLETION_CPU", "31")
    plan = cpu_affinity.plan_cpu_moe_affinity(settings)
    assert plan.numa_nodes == (2,)
    assert plan.compute_cpus == ((21, 23),)
    assert plan.host_weight_numa_node == 4
    assert (plan.poll_cpu, plan.completion_cpu, plan.dma_cpu) == (22, 31, 24)


def test_topology_keeps_manual_numa_and_role_overrides(topology, npu_locality, monkeypatch):
    settings = topology[0]
    settings.numa_nodes = (2,)
    settings.host_weight_numa_node = 4
    settings.native_dispatch_dma_cpu = 20
    monkeypatch.setenv("KT_MOE_POLL_CPU", "31")
    monkeypatch.setenv("KT_MOE_COMPLETION_CPU", "23")
    plan = cpu_affinity.plan_cpu_moe_affinity(settings, resolve_physical_npu=lambda: 7)
    assert (plan.numa_nodes, plan.host_weight_numa_node) == ((2,), 4)
    assert plan.compute_cpus == ((21, 24, 25),)
    assert (plan.dma_cpu, plan.poll_cpu, plan.completion_cpu) == (20, 31, 23)


def test_explicit_thread_count_can_choose_another_node(topology):
    settings, _, _ = topology
    settings.cpuinfer_threads = 4
    plan = cpu_affinity.plan_cpu_moe_affinity(settings)
    assert plan.numa_nodes == (2,)
    assert len(plan.compute_cpus[0]) == 4

    settings.numa_nodes = (4,)
    with pytest.raises(ValueError, match="needs 4 compute cores; NUMA node 4 has 3"):
        cpu_affinity.plan_cpu_moe_affinity(settings)


def test_rejects_explicit_poll_outside_affinity(topology, monkeypatch):
    monkeypatch.setenv("KT_MOE_POLL_CPU", "6")
    with pytest.raises(ValueError, match="poll CPU 6 is outside"):
        cpu_affinity.plan_cpu_moe_affinity(topology[0])


def test_explicit_dma_can_use_another_cpu_in_effective_cpuset(topology, monkeypatch):
    settings, _, _ = topology
    settings.native_dispatch_dma_cpu = 40
    monkeypatch.setattr(cpu_affinity, "eligible_affinity_cpus", lambda cpus: cpus & {40})
    plan = cpu_affinity.plan_cpu_moe_affinity(settings)
    assert plan.dma_cpu == 40
    assert plan.compute_cpus == ((20, 21, 23, 24, 25),)


def test_rejects_explicit_service_cores_on_same_physical_core(topology, monkeypatch):
    monkeypatch.setenv("KT_MOE_POLL_CPU", "20")
    monkeypatch.setenv("KT_MOE_COMPLETION_CPU", "22")
    with pytest.raises(ValueError, match="completion CPU 22 shares a physical core"):
        cpu_affinity.plan_cpu_moe_affinity(topology[0])


def test_rejects_unavailable_weight_numa_node(topology):
    settings, _, _ = topology
    settings.host_weight_numa_node = 1
    with pytest.raises(ValueError, match=r"NUMA nodes \[1\]"):
        cpu_affinity.plan_cpu_moe_affinity(settings)


def test_rejects_insufficient_compute_and_service_cores(topology):
    settings, available, _ = topology
    available.intersection_update({20, 21, 22, 23})
    with pytest.raises(ValueError, match="has 0 available after reserving"):
        cpu_affinity.plan_cpu_moe_affinity(settings)
