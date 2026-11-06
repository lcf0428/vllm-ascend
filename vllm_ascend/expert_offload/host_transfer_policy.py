"""CPU-owned hybrid admission; LRC chooses victims, not compute ownership."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

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
            n = len(missing) * self.ratio.numerator // self.ratio.denominator
            n = min(n, self.config.balance_max_transfers)
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
