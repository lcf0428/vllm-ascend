# Exclusive Host/NPU expert storage

Enable the first version with:

```json
{
  "expert_offload_config": {
    "expert_offload": true,
    "exclusive_expert_storage": true,
    "num_device_experts": 32,
    "num_device_layers": 2,
    "h2d_backend": "torch",
    "cache_policy_enabled": true
  }
}
```

Keep `cpu_moe.enabled`, `enable_multi_card`, `expert_prefetch_enabled`, and
`prefill_split_gmm` disabled. The initial implementation requires TP=PP=DP=1,
the existing v1 runner, and quantized expert offload. Native shared MXFP4 CPU
weights are excluded because CPU MoE is excluded; the existing NPU-only
MXFP4 weight-paging representation remains supported when Host/device quant
dtypes match. Existing expert-offload restrictions still apply.

## Ownership and exchange

For each layer with E experts and K occupied device slots, Host retains E-K
expert bundles. One additional Host bundle per distinct layout is shared
across layers as a rotating spare. A bundle includes w13, w2, and all stored
scales, offsets and scale biases. Device parameter addresses remain fixed.

Each miss performs D2H(victim -> spare), waits, performs H2D(incoming -> device
slot), waits, and commits ownership. The incoming Host bundle becomes the
next spare. There is no NPU staging and no extra Host-to-Host copy in decode.
Slot buffers are allocated before graph capture. Runtime descriptors resolve
the current Host owner; cached descriptors never retain old expert addresses.

Startup places experts directly: initial resident experts [0,K) go through the
original NPU weight loader and receive no Host weight or quantization buffers.
Only the remaining E-K experts allocate pinned Host buffers. Checkpoint copies
are synchronous in this mode, avoiding queued owned clones; gate/up scale
shards write directly into their final slices instead of retaining first-shard
copies. Host weight format conversion runs one expert at a time, so it never
stacks a complete layer on CPU or NPU. Scale/bias conversion replaces each
expert independently. The pool adopts those allocations without repinning or
constructing resident backups. Hot preload uses bounded exchanges, and Prefill
initialization reads resident experts D2D and missing experts H2D.

`[EXPERT-EXCLUSIVE] direct_placement=true conversion_batch_experts=1` reports
live pool bytes including exchange buffers. This removes the offloader's full
Host checkpoint duplicate and full-layer weight-conversion temporary. It does
not bound the upstream checkpoint loader's input tensors/page cache, NPU
allocator reservations, other model components, or the complete Prefill pool.
Measure RSS and NPU peak allocations separately on the serving machine.

## Prefill and graph execution

The original complete Prefill pool remains allocated. Resident experts are
copied D2D from decode slots; missing experts are copied H2D from current Host
owners. This automatically enables `prefill_delta_load`. Prefill copies are
synchronous in this version; the background layer-loading pipeline is disabled
so a Host source cannot be recycled while a Prefill reader is outstanding.
Prefill does not change long-lived ownership. Its temporary duplicate weights
are additional memory, so total memory is one persistent copy plus the full
Prefill pool and exchange buffers.

Small-batch decode keeps the existing ACL graph Host callback mechanism.
Every replay observes fresh Top-K IDs, exchanges experts on the load stream,
and publishes the new mapping before dispatch/GMM. Eager transfer errors
propagate. A replay callback failure terminates the worker: returning from a
partially destructive swap could otherwise silently execute incorrect weights.
Large-batch Prefill remains eager under the existing threshold/capture rules.

## Validation

Dependency-free orchestration tests:

```bash
python tests/ut/expert_offload/test_exclusive_pool.py -v
```

Small NPU startup regression without loading a checkpoint. It checks that
resident MXFP4 slots already converted by the quantization loader retain
their allocations, strides and format when the manager finalizes them.
Both numeric format 29 and named `FRACTAL_NZ_C0_16` are recognized:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_exclusive_expert_graph.py::test_mxfp4_finalization_preserves_real_nz_slots
```

Hardware tests also cover eager and graph-replayed swaps, including quant metadata:

```bash
pytest -v tests/e2e/nightly/single_node/ops/singlecard_ops/test_exclusive_expert_graph.py
```

Before deployment, compare the same quantized model with the option disabled
and enabled in eager and graph modes. Alternate prompts to force repeated
evictions and reloads; include multiple prefill/decode transitions. Compare
outputs and live Host allocation bytes, measure startup peak separately, and
profile both D2H and H2D. The swap tests use plain byte tensors; the NZ startup
test checks format preservation only. Actual NZ/MXFP4 model execution must
additionally validate physical layout roundtrips.
