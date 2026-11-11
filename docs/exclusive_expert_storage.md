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

Each layer's misses form one transaction. It prepares all descriptors, submits
ordered D2H(victim -> spare), H2D(incoming -> device slot) pairs on the same
load stream, refreshes derived scales, and waits once before committing
ownership. The incoming Host bundle becomes the next spare. There is no NPU
staging and no extra Host-to-Host copy in decode.
Slot buffers are allocated before graph capture. Runtime descriptors resolve
the current Host owner; cached descriptors never retain old expert addresses.

The batch validates unique device slots and incoming IDs; every incoming
expert must be Host-owned at batch entry. For consecutive swaps A1/B1 and
A2/B2, the queue is `D2H(A1 -> spare), H2D(B1 -> slot1),
D2H(A2 -> old Host B1), H2D(B2 -> slot2)`. All fields of each direction are
submitted before the next direction. Do not group all D2H ahead of H2D:
that would overwrite B1 before it is read. Same-stream ordering preserves
each victim before its device slot is overwritten, and consumes each incoming
Host bundle before that allocation is reused as the next spare.

The pool lock covers planning, submission, completion and ownership commit,
including layers sharing an exchange spare. Preparation failures submit no
work. Submission/completion failures poison the pool, attempt to drain queued
work, and leave committed ownership maps unchanged; the worker must stop
because physical contents may already have changed. Successful nonempty
decode batches use one completion barrier instead of two per miss plus an
outer barrier. The graph's mapping upload remains after the callback returns.

Startup places experts directly: initial resident experts [0,K) go through the
original NPU weight loader and receive no Host weight or quantization buffers.
Only the remaining E-K experts allocate pinned Host buffers. Checkpoint copies
are synchronous by default, avoiding queued owned clones; gate/up scale
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

## Next-layer prediction and runtime Host copies

This optional mode predicts the next MoE layer's demand, selects likely
eviction victims, and prepares Host copies without moving predicted incoming
experts into NPU slots. Add these fields to `expert_offload_config`:

```json
{
  "exclusive_expert_storage": true,
  "exclusive_host_prefetch_enabled": true,
  "exclusive_host_prefetch_num": 2,
  "exclusive_host_prefetch_tokens": 1,
  "exclusive_host_cache_size_mb": 1024,
  "exclusive_host_prefetch_d2h": true,
  "expert_prefetch_enabled": false
}
```

The Host mode defaults off, its cache budget defaults to 1024 MiB, and early
D2H defaults off. With early D2H off, the cache retains existing Host sources
after reactive H2D; there is no additional CPU-to-CPU weight copy. It has a
warm-up period: initial NPU-only experts still require D2H on their first
eviction unless an early backup is enabled. Predictions protect likely
victim copies from cache replacement. With early D2H on, missing victim
copies are downloaded asynchronously ahead of the next layer's demand.

The cache is a fixed set of pinned expert bundles allocated before capture,
within the global byte budget, round-robin across distinct expert layouts.
Unused capacity smaller than a bundle remains unallocated. The budget must
fit at least one complete bundle; a layout without cache slots falls back to
ordinary exchange. Cache slots, required Host slots and the exchange spare
rotate roles without allocating new weight memory at runtime. Total persistent
weight storage is approximately one model copy plus this bounded redundancy
and the exchange spares; the complete Prefill pool is still additional.
Gate-weight prediction copies and small pinned route buffers are outside the
expert-cache budget.

The existing heuristic next-layer gate prediction and hash routing are used
(not next-token prediction or the CPU/NPU hybrid prefetch driver). Prediction
is staged after current-layer reactive loading and before its dispatch/GMM.
`exclusive_host_prefetch_tokens` controls sampled token rows (default 1,
bounded by available rows); `exclusive_host_prefetch_num` caps predicted
misses/victim candidates (default 2, bounded by the model's routing top-k).
Both must be positive integers and apply only when
`exclusive_host_prefetch_enabled` is true. They are independent of the legacy
`expert_prefetch_num` and `expert_prefetch_tokens`: changing those legacy
settings does not affect Host prefetch. `expert_prefetch_enabled` remains
false in exclusive mode because it controls prefetch into NPU slots. LRC
selects cold victims when enabled; otherwise physical-slot order is used.
All predicted residents are protected. Actual routing and victim selection
remain authoritative and are not changed to force a cache hit.

On each replay, dedicated per-target pinned prediction buffers feed a Host
callback. The callback reserves backup slots and queues only D2H on the same
load stream as reactive exchanges, then returns without waiting. That allows
backup DMA to overlap current-layer compute. Cached bytes may still be in
flight: they are valid only for subsequent operations ordered on that stream,
never for unsynchronized CPU reads. Reusing a pending backup's memory also
happens only through later operations on that stream. No separate event,
compute-stream synchronization, NPU staging allocation, or residency-map
change is introduced by prediction.

At actual eviction, a victim with a Host copy skips reactive D2H: its copy
becomes the required Host source and the incoming expert's old Host bundle
becomes a resident backup. On a miss, ordinary ordered D2H/H2D exchange is
used; an available cache slot can replace the rotating spare so the incoming
source can be retained. Only redundant resident copies may be discarded.
The pool lock and batch transaction cover both cache and ownership changes.
Wrong predictions may waste bandwidth but cannot overwrite an unpreserved
expert. All-hit layers need not wait for speculative backup reads before GMM.
Submission/completion failures terminate the worker instead of reusing an
uncertain copy. This assumes inference weights are immutable, as in the
existing offload path.

`[EXCLUSIVE-HOST-CACHE]` periodically reports counters using
`cache_stats_log_interval`, and reports final counters at close:

- `cache_hits` / `saved_bytes`: reactive D2H avoided by cached victim copies.
- `backup_d2h` / `backup_bytes`: speculative D2H that was performed early.
- `reactive_d2h` / `reactive_bytes`: demand-time D2H still required.
- `retained`, `evicted`, `predictions`, `candidates`: cache/prediction activity.

Early backup hides transfer latency only when computation overlaps it; it
does not inherently reduce total D2H bytes. Compare `backup_bytes +
reactive_bytes`, cache hit rate, TPOT and profiler overlap against cache-only
and disabled runs. No speedup is assumed without hardware measurement.

## Optional parallel checkpoint loading

The checkpoint-loading pool described below is separate from the runtime
Host-copy cache; neither option enables NPU weight staging.

To overlap checkpoint iteration with parallel transpose/copies into the final
Host buffers, add these fields to `expert_offload_config`:

```json
{
  "exclusive_expert_storage": true,
  "exclusive_load_pool_enabled": true,
  "exclusive_load_pool_size_mb": 1024
}
```

`exclusive_load_pool_enabled` defaults to `false`. When enabled it requires
exclusive storage. `exclusive_load_pool_size_mb` is a positive integer in MiB
(1 MiB = 1,048,576 bytes), default 1024. It limits one shared, startup-only CPU
staging arena across all layers, not a per-layer allocation. The arena is
ordinary CPU memory; the final expert Host buffers remain pinned.

The first eligible nonresident shard lazily allocates the configured arena.
The loader copies checkpoint data synchronously into a leased arena range,
then returns while one of the existing 32 workers writes the final Host
destination. Ranges are aligned to 64 bytes and reused only after the writer
completes. Worker tasks do not retain checkpoint tensors or mmap views. This
also applies to gate/up scale shards, which write disjoint final slices.
Resident experts bypass Host staging entirely.

If the arena is full or fragmented, the producer waits for completed writers
and coalesces free ranges. At most 128 tasks are pending. A single shard too
large for the configured arena is copied synchronously after prior writers
finish, without allocating an oversized temporary. Thus a very small pool
can reduce parallelism or cause synchronous fallbacks.

Loading finalization waits for all writes, propagates worker errors, releases
the arena, and restores torch intra-op threads before NZ conversion, Prefill
pool initialization, and graph capture. The runtime exchange spare pool is
separate and unchanged. This option accelerates checkpoint-to-Host loading;
it does not batch or parallelize the later per-expert NZ conversion.

`[EXCLUSIVE-LOAD]` logs report capacity, peak leased bytes, staged/fallback
shard counts and backpressure wait time. The arena allocation equals its
configured capacity once created; peak leased bytes are not total RSS. The
budget excludes checkpoint mappings/page cache, final Host/NPU weights,
thread stacks, packing/scale-operation temporaries, allocator retention and
Prefill storage. Synchronous loading now also records Host load-phase wall
time rather than reporting zero.

On the serving machine, compare both modes with the same checkpoint and
thread settings, including the final drain and NZ-conversion phases. A CPU
transpose-copy microbenchmark (no checkpoint or model required) is available:

```bash
python benchmarks/benchmark_exclusive_load_pool.py --pool-size-mb 1024
```

It checks copied values and reports synchronous/pooled wall time including
drain, copy throughput, and peak staging leases. It excludes SSD I/O and NZ
conversion, so full-model startup timing and RSS still need separate checks.

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
python tests/ut/expert_offload/test_load_buffer_pool.py -v
python tests/ut/expert_offload/test_exclusive_host_prefetch.py -v
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
pytest -v tests/e2e/nightly/single_node/ops/singlecard_ops/test_exclusive_load_pool.py
```

Before deployment, compare the same quantized model with the option disabled
and enabled in eager and graph modes. Alternate prompts to force repeated
evictions and reloads; include multiple prefill/decode transitions. Compare
outputs and live Host allocation bytes, measure startup peak separately, and
profile both D2H and H2D. The swap tests use plain byte tensors; the NZ startup
test checks format preservation only. Actual NZ/MXFP4 model execution must
additionally validate physical layout roundtrips.
