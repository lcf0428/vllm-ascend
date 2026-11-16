"""Attach CPU MoE after the shared NZ buffers have been prepared."""

from __future__ import annotations

from typing import Any

from vllm_ascend.expert_offload.hybrid_executor import (
    CPUExpertBackend,
    attach_hybrid_expert_executor,
    get_hybrid_expert_executor,
)

from .kt_kernel_backend import KtKernelBackendConfig, KtKernelCPUExpertBackend


def model_layer_index(layer_name: str) -> int:
    """Resolve the model layer index, which may differ from the MoE ordinal."""
    from vllm.model_executor.models.utils import extract_layer_index

    try:
        return int(extract_layer_index(layer_name))
    except (AssertionError, ValueError, IndexError) as exc:
        raise ValueError(
            f"Cannot resolve the CPU MoE model layer index from {layer_name!r}"
        ) from exc


def maybe_attach_cpu_expert_backend(
    *,
    routed_experts: Any,
    layer_name: str,
    moe_config: Any,
    cpu_moe_config: Any,
    enable_multi_card: bool,
    tp_size: int,
    native_weights=None,
) -> CPUExpertBackend | None:
    """Attach the CPU backend for a registered MoE layer."""
    if not cpu_moe_config.enabled:
        return None

    existing = get_hybrid_expert_executor(routed_experts)
    if existing is not None:
        return existing.backend

    model_layer_idx = model_layer_index(layer_name)

    # Hybrid supports single-card decode and DSpark verification.  Refuse the rest loudly rather than
    # attaching a backend whose routes would never reach it.
    if enable_multi_card or tp_size != 1 or moe_config.ep_size > 1:
        raise ValueError(
            "expert_offload_config.cpu_moe supports single-card decode only, "
            f"but layer {model_layer_idx} runs with enable_multi_card="
            f"{enable_multi_card}, tp_size={tp_size}, "
            f"ep_size={moe_config.ep_size}"
        )

    affinity = getattr(cpu_moe_config, "startup_affinity", None)
    if affinity is None:
        raise RuntimeError("Resolve CPU MoE affinity before attaching the CPU backend")
    config = KtKernelBackendConfig(
        layer_idx=model_layer_idx,
        num_experts=moe_config.num_logical_experts,
        top_k=moe_config.experts_per_token,
        hidden_size=moe_config.hidden_dim,
        intermediate_size=moe_config.intermediate_size_per_partition,
        cpuinfer_threads=sum(map(len, affinity.compute_cpus)),
        threadpool_count=cpu_moe_config.threadpool_count,
        max_num_tokens=cpu_moe_config.max_num_tokens,
        numa_nodes=affinity.numa_nodes,
        compute_cpus=affinity.compute_cpus,
        poll_cpu=affinity.poll_cpu,
        completion_cpu=affinity.completion_cpu,
    )
    backend = KtKernelCPUExpertBackend(config, native_weights=native_weights)
    # Both sides take the model layer index: submit() cross-checks the index it
    # is handed against config.layer_idx and raises on a mismatch.
    attach_hybrid_expert_executor(routed_experts, backend, model_layer_idx)
    return backend


__all__ = ["maybe_attach_cpu_expert_backend", "model_layer_index"]


def attach_or_defer_cpu_expert_backend(manager, **kwargs):
    """Queue eligible layers until their shared DMA storage is verified."""
    if not kwargs["cpu_moe_config"].enabled:
        return None
    if getattr(manager, "_cpu_weights_finalized", False):
        raise RuntimeError("CPU MoE layers must register before NZ weight finalization")
    pending = getattr(manager, "_pending_cpu_backend_attachments", None)
    if pending is None:
        pending = manager._pending_cpu_backend_attachments = []
    if not any(item["routed_experts"] is kwargs["routed_experts"] for item in pending):
        pending.append(kwargs)


def finalize_deferred_cpu_backends(manager):
    """Attach all registered MoE layers to their verified NZ weight owners."""
    if getattr(manager, "_cpu_weights_finalized", False):
        return 0
    pending = getattr(manager, "_pending_cpu_backend_attachments", [])
    if {id(item["routed_experts"]) for item in pending} != {id(layer) for layer in manager.moe_layers}:
        raise RuntimeError("CPU MoE requires a pending backend for every registered MoE layer")
    if not getattr(manager, "_nz_cpu_layout_ready", False):
        raise RuntimeError("NZ CPU cannot attach before physical weight conversion")

    from kt_kernel.utils.nz_mxfp4 import NZWeights

    for kwargs in pending:
        layer_idx = manager.moe_layers.index(kwargs["routed_experts"])
        if layer_idx not in manager._nz_cpu_layout_verified:
            raise RuntimeError("NZ CPU requires verified physical NZ source bytes")
        native_weights = NZWeights(
            tuple(manager.w13_weights_cpu[layer_idx]),
            tuple(manager.w2_weights_cpu[layer_idx]),
            tuple(manager.scale_cpu_buffers["w13_weight_scale"][layer_idx]),
            tuple(manager.scale_cpu_buffers["w2_weight_scale"][layer_idx]),
        )
        maybe_attach_cpu_expert_backend(**kwargs, native_weights=native_weights)
    count = len(pending)
    pending.clear()
    manager._cpu_weights_finalized = True
    return count
