"""Fixed Host allocations with exclusive Host/device expert ownership.

Independent of the accelerator runtime: callers supply synchronous transfer
operations. Each layout has one rotating spare bundle shared across layers.
Allocation happens before capture; swap never allocates weight storage.
"""

import threading


class ExclusiveExpertPool:
    def __init__(self):
        self.host = {}
        self.resident = {}
        self.spares = {}
        self.layouts = {}
        self.failed = False
        self.lock = threading.RLock()

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
        if self.failed:
            raise RuntimeError("Exclusive expert pool is poisoned after a transfer failure")
        return self.host[layer_idx, eid]

    def swap(self, layer_idx, slot, incoming, download, upload):
        """Save victim, upload incoming, then atomically publish ownership.

        download(bundle) must finish D2H before returning; upload(bundle) must
        finish H2D before returning. A failure poisons the pool: callers must
        stop execution rather than publish a mapping for incomplete weights.
        """
        with self.lock:
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
