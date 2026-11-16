"""CPU MoE topology discovery resolves the bound device before reading locality."""

import subprocess
from unittest.mock import Mock

import pytest

from vllm_ascend.expert_offload import npu_topology


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    for node, cpus in ((2, "20-25"), (4, "30-35")):
        path = tmp_path / f"node{node}"
        path.mkdir()
        (path / "cpulist").write_text(cpus)
    monkeypatch.setattr(npu_topology, "_NUMA_ROOT", tmp_path)


def test_physical_id_table_uses_resolved_id(sysfs, monkeypatch):
    command = Mock(return_value="Phy-ID0    Phy-ID7    CPU Affinity\n"
                                "Phy-ID0    X    SYS    20-25\n"
                                "Phy-ID7    SYS    X    30-31, 34-35\n")
    monkeypatch.setattr(npu_topology, "_npu_smi", command)
    assert npu_topology.discover_npu_numa(lambda: 7) == (4, set(range(30, 36)))
    command.assert_called_once_with("-t", "topo")


@pytest.mark.parametrize(
    "mapping",
    [
        "NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID  Chip Name\n"
        "2   0   0   7   Ascend\n",
        "NPU ID  Slot ID  Chip ID  Chip Phy-ID  Chip Name\n"
        "0   0   0   7   Ascend950DT\n",
    ],
)
def test_maps_legacy_table_to_correct_id(sysfs, monkeypatch, mapping):
    command = Mock(side_effect=[
        "NPU0    NPU7   CPU Affinity\nNPU0    X    SYS    30-35\nNPU7    SYS    X    20-25\n",
        mapping,
    ])
    monkeypatch.setattr(npu_topology, "_npu_smi", command)
    assert npu_topology.discover_npu_numa(lambda: 7) == (4, set(range(30, 36)))


def test_older_mapping_without_physical_column_uses_remap(sysfs, monkeypatch):
    command = Mock(side_effect=[
        "NPU0    CPU Affinity\nNPU0    X    30-35\n",
        "NPU ID  Chip ID  Chip Logic ID  Chip Name\n2  0  0  Ascend\n",
        "Chip Physical ID : 7\nChip Logic ID : 0\nNPU ID : 2\n",
    ])
    monkeypatch.setattr(npu_topology, "_npu_smi", command)
    assert npu_topology.discover_npu_numa(lambda: 7) == (4, set(range(30, 36)))
    assert command.call_args.args == ("-t", "phyid-remap", "-p", "7")


@pytest.mark.parametrize("cpus", ["N/A", "20-25,30-35", "80-85"])
def test_ambiguous_or_missing_cpu_affinity_has_no_preference(sysfs, monkeypatch, cpus):
    monkeypatch.setattr(npu_topology, "_npu_smi", lambda *_: f"Phy-ID7  CPU Affinity\nPhy-ID7  X  {cpus}\n")
    assert npu_topology.discover_npu_numa(lambda: 7) is None


def test_failed_device_translation_does_not_guess_a_card(monkeypatch):
    query = Mock()
    monkeypatch.setattr(npu_topology, "_npu_smi", query)
    assert npu_topology.discover_npu_numa(Mock(side_effect=RuntimeError("mapping unavailable"))) is None
    query.assert_not_called()


def test_topology_command_timeout_allows_capacity_fallback(monkeypatch):
    command = Mock(side_effect=subprocess.TimeoutExpired("npu-smi", 3))
    monkeypatch.setattr(npu_topology.subprocess, "run", command)
    assert npu_topology.discover_npu_numa(lambda: 7) is None
    assert command.call_args.kwargs["timeout"] == 3
