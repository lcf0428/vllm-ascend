# SPDX-License-Identifier: Apache-2.0

"""Read the bound NPU's NUMA affinity for CPU MoE placement."""

import logging
import os
import re
import subprocess
from pathlib import Path

from .host_weight_allocator import parse_cpu_list

_LOG = logging.getLogger(__name__)
_NUMA_ROOT = Path("/sys/devices/system/node")


def _npu_smi(*args: str) -> str:
    try:
        return subprocess.run(
            ["npu-smi", "info", *args],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "LC_ALL": "C"},
            timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        _LOG.info("CPU MoE topology query %s unavailable: %s", args, error)
        return ""


def _legacy_npu_id(physical_id: int) -> int | None:
    # Legacy NPU<n> tables use Chip Logic ID when present, like the existing
    # Ascend topology consumer. On 950 mappings without it, use NPU ID.
    rows = _npu_smi("-m").splitlines()
    for i, row in enumerate(rows):
        fields = re.split(r"\s{2,}", row.strip())
        if "NPU ID" not in fields:
            continue
        physical_field = next(
            (j for j, name in enumerate(fields) if name.startswith(("Chip Phy-ID", "Chip Physical ID"))),
            None,
        )
        if physical_field is None:
            break
        id_field = fields.index("Chip Logic ID") if "Chip Logic ID" in fields else fields.index("NPU ID")
        matches = set()
        for entry in rows[i + 1:]:
            values = entry.split()
            if (
                len(values) > max(id_field, physical_field)
                and values[physical_field] == str(physical_id)
                and values[id_field].isdigit()
            ):
                matches.add(int(values[id_field]))
        if matches:
            return next(iter(matches)) if len(matches) == 1 else None
        break

    # Older mapping tables may omit the physical chip column.
    remap = _npu_smi("-t", "phyid-remap", "-p", str(physical_id))
    ids = {name.strip(): int(value) for name, value in re.findall(r"^\s*([^:\n]+):\s*(\d+)\s*$", remap, re.M)}
    if ids.get("Chip Physical ID") == physical_id:
        return ids.get("Chip Logic ID", ids.get("NPU ID"))
    return None


def _affinity_cpus(topo: str, label: str) -> set[int]:
    rows = []
    for line in topo.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0] == label and not re.search(r"CPU\s+Affinity", parts[1]):
            rows.append(parts[1])
    if len(rows) != 1:
        return set()
    cpus = re.search(r"(?:^|\s)(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\s*$", rows[0])
    return parse_cpu_list(cpus[1]) if cpus else set()


def discover_npu_numa(resolve_physical_npu) -> tuple[int, set[int]] | None:
    """Return the NPU's unambiguous NUMA node and its full host CPU list."""
    try:
        physical_id = resolve_physical_npu()
        topo = _npu_smi("-t", "topo")
        if not re.search(r"CPU\s+Affinity", topo):
            return None
        if re.search(r"^\s*Phy-ID\d+\s", topo, re.M):
            cpus = _affinity_cpus(topo, f"Phy-ID{physical_id}")
        else:
            topo_id = _legacy_npu_id(physical_id)
            cpus = _affinity_cpus(topo, f"NPU{topo_id}") if topo_id is not None else set()
        if not cpus:
            return None
        for path in _NUMA_ROOT.glob("node[0-9]*"):
            node_cpus = parse_cpu_list((path / "cpulist").read_text())
            if cpus <= node_cpus:
                return int(path.name[4:]), node_cpus
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as error:
        _LOG.info("CPU MoE could not resolve the bound NPU's NUMA node: %s", error)
    return None
