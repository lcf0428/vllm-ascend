"""CPU-owned hybrid admission; LRC chooses victims, not compute ownership."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import lcm

from .frequency_transfer_policy import (
    FREQUENCY_DECAY,
    FREQUENCY_MIN_SCORE,
    FREQUENCY_REPLACEMENT_RATIO,
)


@dataclass(frozen=True)
class HostTransferPlan:
    mapping: tuple[int, ...]
    loads: tuple[tuple[int, int], ...]  # (slot, logical expert)
    cpu_ids: tuple[tuple[int, ...], ...]
    cpu_weights: tuple[tuple[float, ...], ...]


class HostTransferPolicy:
    """One planner per layer; only the input-poll producer mutates its state."""

    def __init__(self, *, experts, slots, config, lrc=None, layer_idx=0):
        self.experts, self.slots = experts, slots
        self.config, self.lrc, self.layer_idx = config, lrc, layer_idx
        self.scores = [0.0] * experts
        cpu = Fraction(str(config.balance_cpu_expert_us))
        copy = Fraction(str(config.balance_copy_us))
        self.ratio = cpu / (cpu + copy)
        fixed = Fraction(str(getattr(config, "balance_npu_fixed_us", 0.0)))
        scale = lcm(cpu.denominator, copy.denominator, fixed.denominator)
        self.cpu_unit = int(cpu * scale)
        self.copy_unit = int(copy * scale)
        self.npu_fixed_unit = int(fixed * scale)

    def plan(self, ids, weights, mapping):
        if len(mapping) != self.experts or len(ids) != len(weights):
            raise ValueError("Host route/mapping shape mismatch")
        owners = {}
        for eid, slot in enumerate(mapping):
            if slot < -1 or slot >= self.slots or (slot >= 0 and slot in owners):
                raise ValueError("Invalid or duplicate resident slot")
            if slot >= 0:
                owners[slot] = eid
        rows, score_rows = [], []
        for row, scores in zip(ids, weights):
            if len(row) != len(scores):
                raise ValueError("Host route/weight width mismatch")
            rows.append([int(e) if 0 <= int(e) < self.experts and w != 0 else -1 for e, w in zip(row, scores)])
            score_rows.append([float(w) if e >= 0 else 0.0 for e, w in zip(rows[-1], scores)])
        counts = Counter(e for row in rows for e in row if e >= 0)
        protected = set(counts)
        self.scores = [s * FREQUENCY_DECAY + counts[e] for e, s in enumerate(self.scores)]
        if self.lrc is not None:
            self.lrc.observe(self.layer_idx, rows, router_scores=score_rows)
        missing = [e for e in counts if mapping[e] < 0]
        free = [s for s in range(self.slots) if s not in owners]
        cold = [e for e in owners.values() if e not in protected]
        if self.lrc is not None:
            cold = self.lrc.choose_victims(self.layer_idx, owners, protected, len(cold))
        else:
            cold.sort(key=lambda e: (self.scores[e], e))
        targets = free + [mapping[e] for e in cold]
        policy = self.config.transfer_policy
        if policy == "load_balance":
            missing.sort(key=lambda e: (-counts[e], -self.scores[e], e))
            # Each expert-token pair is CPU work; each expert weight is copied
            # once, regardless of how many tokens selected it. Evaluate all
            # frequency-ranked prefixes and minimize the parallel makespan.
            remaining_routes = sum(counts[e] for e in missing)
            cpu_unit, copy_unit = self.cpu_unit, self.copy_unit
            # One whole-layer NPU compute estimate, including the k=0 baseline.
            best_cost = max(remaining_routes * cpu_unit, self.npu_fixed_unit)
            n = 0
            limit = min(len(missing), len(targets), self.config.balance_max_transfers)
            for copied in range(1, limit + 1):
                remaining_routes -= counts[missing[copied - 1]]
                cost = max(remaining_routes * cpu_unit, copied * copy_unit + self.npu_fixed_unit)
                # Equal makespan keeps the smaller transfer count.
                if cost < best_cost:
                    best_cost, n = cost, copied
        elif policy == "frequency":
            # Stable input order breaks equal-frequency ties, as on device.
            missing.sort(key=lambda e: -self.scores[e])
            n = 1
        else:
            n = 1
        loads = []
        updated = list(mapping)
        for eid, slot in zip(missing[:n], targets):
            victim = owners.get(slot)
            if policy == "frequency":
                demand = self.scores[eid]
                old = self.scores[victim] if victim is not None else 0.0
                if demand < FREQUENCY_MIN_SCORE or demand <= old * FREQUENCY_REPLACEMENT_RATIO:
                    continue
            if victim is not None:
                updated[victim] = -1
            updated[eid] = slot
            loads.append((slot, eid))
        cpu_ids, cpu_weights = [], []
        for row, scores in zip(rows, score_rows):
            mask = [e >= 0 and updated[e] < 0 for e in row]
            cpu_ids.append(tuple(e if owned else -1 for e, owned in zip(row, mask)))
            cpu_weights.append(tuple(w if owned else 0.0 for w, owned in zip(scores, mask)))
        return HostTransferPlan(tuple(updated), tuple(loads), tuple(cpu_ids), tuple(cpu_weights))
