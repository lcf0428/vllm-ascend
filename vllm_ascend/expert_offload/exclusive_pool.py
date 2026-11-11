"""Fixed Host allocations with exclusive Host/device expert ownership.

Independent of the accelerator runtime: callers supply ordered transfer
submission and completion operations. Each layout has one rotating spare bundle shared across layers.
Allocation happens before capture; swap never allocates weight storage.
"""

import threading
from collections import OrderedDict
from typing import NamedTuple


class ExclusiveSwap(NamedTuple):
    slot: int
    incoming: int
    victim: int
    source: dict
    destination: dict
    download: bool = True


class ExclusiveExpertPool:
    def __init__(self):
        self.host = {}
        self.resident = {}
        self.spares = {}
        self.layouts = {}
        self.failed = False
        self.lock = threading.RLock()
        # Redundant resident copies, valid in the caller's ordered DMA stream.
        # An early D2H can still be pending: never read payloads on the CPU.
        self.shadows = OrderedDict()
        self.shadow_free = {}
        self.shadow_targets = set()
        self.shadow_capacity_bytes = 0
        self.shadow_stats = dict(predictions=0, candidates=0, backup_d2h=0, cache_hits=0,
                                 reactive_d2h=0, retained=0, evicted=0, backup_bytes=0, saved_bytes=0,
                                 reactive_bytes=0)

    def configure_shadow_cache(self, capacity_bytes, allocate):
        """Preallocate a bounded cache, round-robin across distinct layouts."""
        with self.lock:
            if self.shadow_capacity_bytes:
                raise RuntimeError("Host shadow cache is already configured")
            templates = {}
            for (idx, _), bundle in self.host.items():
                templates.setdefault(self.layouts[idx], bundle)
            remaining = capacity_bytes
            while True:
                progress = False
                for layout, template in templates.items():
                    size = sum(t.untyped_storage().nbytes() for t in template.values())
                    if size <= 0:
                        raise ValueError("Host shadow cache requires nonempty expert bundles")
                    if size <= remaining:
                        self.shadow_free.setdefault(layout, []).append(allocate(template))
                        remaining -= size
                        self.shadow_capacity_bytes += size
                        progress = True
                if not progress:
                    break
            if not self.shadow_capacity_bytes:
                raise ValueError("exclusive_host_cache_size_mb cannot hold even one expert bundle")

    def _take_shadow_buffer(self, layout, shadows, free, protected):
        if free.get(layout):
            return free[layout].pop()
        for key in list(shadows):
            if key not in protected and self.layouts[key[0]] == layout:
                return shadows.pop(key)
        return None

    def prefetch_backups(self, layer_idx, victims, early_d2h, prepare, enqueue):
        """Reserve victim backups without changing residency or waiting for DMA.

        All payload operations, including later reactive swaps, MUST use the
        same ordered stream. Pending backup ranges can therefore be reused
        only by subsequently queued operations, never by CPU writes.
        """
        with self.lock:
            if self.failed:
                raise RuntimeError("Exclusive expert pool is poisoned after a transfer failure")
            slots = {eid: slot for (idx, slot), eid in self.resident.items() if idx == layer_idx}
            victims = list(dict.fromkeys(eid for eid in victims if eid in slots))
            targets = {(layer_idx, eid) for eid in victims}
            shadows = self.shadows.copy()
            free = {layout: list(bundles) for layout, bundles in self.shadow_free.items()}
            plan = []
            for eid in victims:
                key = (layer_idx, eid)
                if key in shadows:
                    shadows.move_to_end(key)
                elif early_d2h:
                    buffer = self._take_shadow_buffer(self.layouts[layer_idx], shadows, free, targets)
                    if buffer is not None:
                        shadows[key] = buffer
                        plan.append((slots[eid], eid, buffer))
            prepared = prepare(plan)
            try:
                if plan:
                    enqueue(prepared)
            except BaseException:
                self.failed = True
                raise
            self.shadow_stats["predictions"] += 1
            self.shadow_stats["candidates"] += len(victims)
            self.shadow_stats["backup_d2h"] += len(plan)
            self.shadow_stats["backup_bytes"] += sum(
                t.untyped_storage().nbytes() for _, _, b in plan for t in b.values())
            self.shadow_stats["evicted"] += len(set(self.shadows) - set(shadows))
            self.shadows, self.shadow_free, self.shadow_targets = shadows, free, targets

    @staticmethod
    def layout(bundle):
        return tuple((name, tuple(t.shape), t.dtype, t.untyped_storage().nbytes())
                     for name, t in sorted(bundle.items()))

    def add_layer(self, layer_idx, bundles, mapping, allocate_spare=None):
        """Adopt Host-only bundles; resident entries may be None.

        allocate_spare creates one empty bundle per layout, never a resident
        weight backup. Dense bundles remain accepted for existing callers.
        """
        with self.lock:
            if layer_idx in self.layouts:
                raise RuntimeError("Exclusive layer was already initialized")
            template = next(bundle for bundle in bundles if bundle is not None)
            layout = self.layout(template)
            if any(self.layout(bundle) != layout for bundle in bundles if bundle is not None):
                raise ValueError("Exclusive experts must have a uniform layout within a layer")
            slots = [slot for slot in mapping if slot >= 0]
            if len(mapping) != len(bundles) or sorted(slots) != list(range(len(slots))) or not slots:
                raise ValueError("Exclusive pool requires a complete, unique resident slot mapping")
            self.layouts[layer_idx] = layout
            if layout not in self.spares and allocate_spare is not None:
                self.spares[layout] = allocate_spare(template)
            for eid, (bundle, slot) in enumerate(zip(bundles, mapping)):
                if slot < 0:
                    if bundle is None:
                        raise ValueError("Nonresident expert is missing its Host bundle")
                    self.host[layer_idx, eid] = bundle
                else:
                    self.resident[layer_idx, slot] = eid
                    if layout not in self.spares:
                        if bundle is None:
                            raise ValueError("Sparse initialization requires an exchange-slot allocator")
                        self.spares[layout] = bundle

    def source(self, layer_idx, eid):
        with self.lock:
            if self.failed:
                raise RuntimeError("Exclusive expert pool is poisoned after a transfer failure")
            return self.host[layer_idx, eid]

    def swap_batch(self, layer_idx, loads, prepare, enqueue, synchronize):
        """Plan, enqueue in stream order, wait once, then publish ownership.

        Without a cached victim: D2H(slot_i -> spare_i), H2D(source_i -> slot_i).
        With a cached victim, only H2D is required; its Host copy is promoted.
        Without retention spare_(i+1) = source_i; retaining source_i instead
        consumes a free/redundant cache allocation as the next spare. A
        single ordered stream ensures each source's last read finishes before
        its next overwrite. Unique device slots and initially Host-only
        incoming IDs prevent intra-batch cycles.

        prepare(plan) must not enqueue device work. enqueue(prepared) must
        preserve this order for ALL fields and retain their backing storage.
        synchronize() must finish every enqueued copy and derived-data update.
        The lock covers planning through completion, including other layers
        sharing the spare. Readers must not use Host sources concurrently.
        """
        with self.lock:
            if self.failed:
                raise RuntimeError("Exclusive expert pool is poisoned after a transfer failure")
            loads = list(loads)
            if not loads:
                self.shadow_targets = {key for key in self.shadow_targets if key[0] != layer_idx}
                return []
            if len({slot for slot, _ in loads}) != len(loads):
                raise ValueError("Exclusive batch contains duplicate device slots")
            if len({eid for _, eid in loads}) != len(loads):
                raise ValueError("Exclusive batch contains duplicate incoming experts")
            layout = self.layouts[layer_idx]
            spare = self.spares[layout]
            shadows = self.shadows.copy()
            free = {key: list(bundles) for key, bundles in self.shadow_free.items()}
            protected = self.shadow_targets | {(layer_idx, self.resident[layer_idx, slot]) for slot, _ in loads}
            plan = []
            for slot, eid in loads:
                # Resolve every source against committed ownership BEFORE
                # submitting anything. Resident-to-resident moves are invalid.
                source = self.host[layer_idx, eid]
                victim = self.resident[layer_idx, slot]
                backup = shadows.pop((layer_idx, victim), None)
                if backup is not None:
                    # Its earlier D2H, if pending, precedes this H2D on the
                    # same stream. Promote the victim copy; retain the source
                    # as the incoming resident's shadow, exchanging ownership.
                    plan.append(ExclusiveSwap(slot, eid, victim, source, backup, False))
                    shadows[layer_idx, eid] = source
                else:
                    plan.append(ExclusiveSwap(slot, eid, victim, source, spare))
                    replacement = self._take_shadow_buffer(layout, shadows, free, protected)
                    if replacement is not None:
                        shadows[layer_idx, eid] = source
                        spare = replacement
                    else:
                        spare = source
            prepared = prepare(tuple(plan))
            try:
                enqueue(prepared)
                synchronize()
            except BaseException:
                self.failed = True
                # Submission can fail after earlier DMA was queued. Retire it
                # before unwinding descriptors; never publish a partial batch.
                try:
                    synchronize()
                except BaseException:
                    pass
                raise
            for step in plan:
                self.host[layer_idx, step.victim] = step.destination
                del self.host[layer_idx, step.incoming]
                self.resident[layer_idx, step.slot] = step.incoming
            self.spares[layout] = spare
            hits = [step for step in plan if not step.download]
            self.shadow_stats["cache_hits"] += len(hits)
            self.shadow_stats["reactive_d2h"] += len(plan) - len(hits)
            self.shadow_stats["reactive_bytes"] += sum(
                t.untyped_storage().nbytes() for step in plan if step.download for t in step.destination.values())
            self.shadow_stats["retained"] += sum((layer_idx, step.incoming) in shadows for step in plan)
            self.shadow_stats["saved_bytes"] += sum(
                t.untyped_storage().nbytes() for step in hits for t in step.destination.values())
            consumed = {(layer_idx, step.victim) for step in plan}
            self.shadow_stats["evicted"] += len(set(self.shadows) - set(shadows) - consumed)
            self.shadows, self.shadow_free = shadows, free
            self.shadow_targets = {key for key in self.shadow_targets if key[0] != layer_idx}
            return [step.victim for step in plan]

    def swap(self, layer_idx, slot, incoming, download, upload):
        """Save victim, upload incoming, then atomically publish ownership.

        download(bundle) must finish D2H before returning; upload(bundle) must
        finish H2D before returning. A failure poisons the pool: callers must
        stop execution rather than publish a mapping for incomplete weights.
        """
        with self.lock:
            if self.shadow_capacity_bytes:
                raise RuntimeError("Use swap_batch when the Host shadow cache is configured")
            source = self.source(layer_idx, incoming)
            victim = self.resident[layer_idx, slot]
            layout = self.layouts[layer_idx]
            spare = self.spares[layout]
            try:
                download(spare)
                upload(source)
            except BaseException:
                self.failed = True
                raise
            self.host[layer_idx, victim] = spare
            del self.host[layer_idx, incoming]
            self.resident[layer_idx, slot] = incoming
            self.spares[layout] = source
            return victim
