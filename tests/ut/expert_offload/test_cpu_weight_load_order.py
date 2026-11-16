"""CPU MoE attaches every registered layer after validating physical NZ."""
import sys
from types import SimpleNamespace

import pytest

from vllm_ascend.expert_offload.cpu_backend import wiring


@pytest.fixture
def startup(monkeypatch):
    layers = [object(), object()]
    owners = [[object()], [object()]]
    manager = SimpleNamespace(
        moe_layers=layers,
        w13_weights_cpu=owners,
        w2_weights_cpu=owners,
        scale_cpu_buffers={"w13_weight_scale": owners, "w2_weight_scale": owners},
        _nz_cpu_layout_ready=False,
        _nz_cpu_layout_verified={0, 1},
    )
    monkeypatch.setitem(sys.modules, "kt_kernel.utils.nz_mxfp4", SimpleNamespace(NZWeights=lambda *args: args))
    attached = []
    monkeypatch.setattr(wiring, "maybe_attach_cpu_expert_backend", lambda **kwargs: attached.append(kwargs))
    for layer, model_idx in zip(layers, (3, 17)):
        wiring.attach_or_defer_cpu_expert_backend(
            manager, routed_experts=layer, layer_name=f"model.layers.{model_idx}",
            cpu_moe_config=SimpleNamespace(enabled=True),
        )
    return manager, owners, attached


def test_all_registered_layers_borrow_their_nz_storage(startup):
    manager, owners, attached = startup
    with pytest.raises(RuntimeError, match="before physical weight conversion"):
        wiring.finalize_deferred_cpu_backends(manager)
    assert not attached
    manager._nz_cpu_layout_ready = True
    assert wiring.finalize_deferred_cpu_backends(manager) == 2
    assert [entry["layer_name"] for entry in attached] == ["model.layers.3", "model.layers.17"]
    for entry, owner in zip(attached, owners):
        assert entry["native_weights"] == (tuple(owner),) * 4
    assert wiring.finalize_deferred_cpu_backends(manager) == 0


def test_missing_layer_attachment_fails_startup(startup):
    manager, _, attached = startup
    manager._pending_cpu_backend_attachments.pop()
    with pytest.raises(RuntimeError, match="every registered MoE layer"):
        wiring.finalize_deferred_cpu_backends(manager)
    assert not attached


def test_unverified_owner_fails_startup(startup):
    manager, _, attached = startup
    manager._nz_cpu_layout_ready = True
    manager._nz_cpu_layout_verified.clear()
    with pytest.raises(RuntimeError, match="verified physical NZ"):
        wiring.finalize_deferred_cpu_backends(manager)
    assert not attached
    assert not getattr(manager, "_cpu_weights_finalized", False)
