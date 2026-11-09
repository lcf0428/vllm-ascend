# Native CPU MoE graph dispatch

`cpu_moe.native_dispatch=true` replaces the per-layer Python input producer,
submission queue and result-completion worker during captured graph replay.
The default remains false, preserving the original path for rollback.

The input stream copies hidden states, routes and the resident mapping before
publishing a monotonically increasing sequence to coherent pinned host memory.
A C++ thread acquire-loads that sequence, runs the same host admission policy,
masks the CPU routes and invokes a pre-registered CPUInfer task through a
versioned C interface. Task arguments, tensors, events, planners and backends
are retained by the Python registration owner until graphs retire and native
threads join. No Python callable or GIL acquisition occurs in this path.

A separate C++ thread drains CPUInfer and publishes the result-ready value.
Torch NPU 2.10 ExternalEvent uses a device allocation and value wait/write;
its `npu_event` must **not** be passed to `aclrtRecordEvent`. Native completion
uses `aclrtValueWrite(address, 1, 0, stream)`, matching Torch NPU's SyncLaunchStream
implementation. The recorded graph consumes and clears that value.

CPU results and expert weights use independent readiness values and streams.
When admission produces no loads, the mapping is unchanged, so C++ publishes
weight readiness directly without a redundant mapping copy. Non-empty expert
loads are handed to a separate Python DMA worker; they never gate native CPU
submission or CPU completion. Eager/prefill retains its existing execution
path, using the same native planner and shared LRC state.

## Native prefetch and demand-first DMA

Prefetch notifications use the same C++ monitor as CPU compute, with distinct
registered channels and sequence buffers. Graph replay does not call an ACL
Python host callback to start either path. Predicted next-layer routes read
that layer's existing C++ LRC history; only actual demand routes update LRC.
CPU computation, CPU completion and DMA submission remain separate workers.

The main stream joins current-layer demand weights before triggering next-layer
prefetch, then submits the current NPU MoE computation. This ordering keeps
speculative copies behind current demand copies on the DMA stream. A load
builds and submits one expert's weights/scales before preparing the next expert,
rather than constructing every expert before the first copy can start. Publish
mapping and readiness only after every required weight and scale operation.
The next layer waits for its prefetched weights and mapping before using them.
The prefetch cap is `expert_prefetch_num`; use `1` for the measured configuration.

Prefetch readiness owns a 32-byte device flag. Reset, signal and wait consistently
use ACL value operations; the bridge drains Torch's host launch queue before
issuing raw ACL operations. Every real graph capture gets distinct site owners,
including captures that occur outside the Python capture-context flag. This
avoids mixing native zero-copy notifications with Torch external-event record
bookkeeping across eager prewarm and multiple captured shapes.

## Route-weighted load balancing

`load_balance` counts expert-token routes: one expert selected by two tokens is
two CPU work units but one weight transfer. Missing experts are ranked by route
frequency, then candidate transfer prefixes are evaluated with a simple model:

```text
CPU(k) = remaining expert-token routes * balance_cpu_expert_us
NPU(k) = k * balance_copy_us + balance_npu_fixed_us
choose k to minimize max(CPU(k), NPU(k))
```

The fixed NPU estimate applies once per layer, including the no-transfer
baseline. It accounts for NPU compute after weights arrive; it is not multiplied
by the number of experts. The default is zero for backward compatibility; the
measured setup explicitly uses 200 us. LRC chooses eligible eviction victims,
while route frequency chooses which missing experts are worth moving to NPU.
Python and native planning share the same integer-scaled cost comparison and
prefer fewer transfers when costs tie.

## Build and configuration

The matching kt-kernel must export `native_dispatch_api()` and
`native_cpu_handle()`. `kt_dispatch_api.h` defines the versioned C interface;
the functions must be compiled into kt-kernel itself so no private class
layout or hidden C++ symbol is guessed across extensions. Keep the existing
mathematical kernel objects unchanged when adding this interface.

Install this extension from source:

```sh
python -m pip install -e ./csrc/lrc_policy --no-build-isolation --no-deps
# Required additionally when native expert prefetch is enabled:
(cd csrc/lrc_policy && python setup_npu_wait.py build_ext --inplace)
```

The optional prefetch bridge links Torch NPU and CANN; the CPU-only LRC build
remains independent of those libraries. Build it in the same Python/Torch NPU
environment as the service. Third-party kt-kernel sources and binaries are not
vendor-copied here: its versioned API exports above remain a prerequisite.

This implementation requires Linux, coherent x86 pinned host memory, the
value-based Torch NPU external-event API and the matching kt-kernel build.
With a cache policy enabled, use C++ LRC. Admission state is shared between
eager execution and all captured shapes, and protected by a state mutex.

Choose CPUs from the actual NPU/NUMA topology. `KT_MOE_POLL_CPU` controls the
native input/planning thread; `KT_MOE_COMPLETION_CPU` controls the native
completion thread. `cpu_moe.native_dispatch_dma_cpu` controls weight DMA;
its default uses the completion CPU. All must be outside the computation
workers' physical cores, including SMT siblings. The verified 153 setup uses
compute 32–61, poll 62, completion 63 and DMA 127 (a communication-core sibling).

## Validation

`test_native_submission.py` compares repeated native plans against the
reference for all three admission policies, with and without seeded LRC,
and checks in-place route conservation for 32/64-bit IDs and mappings.

`native_dispatch_fixture.cpp` and `native_dispatch_npu_check.py` provide a small
real NPU graph check. The test deliberately holds Python's GIL while waiting
for a native task, checks five exactly-once generations, and verifies that
both independent external values release the captured graph. Run this only
with the intended NPU explicitly selected; then validate actual model outputs
and benchmark with profiling disabled.
