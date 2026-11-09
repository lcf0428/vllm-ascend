"""CPU weights finalize after DMA allocation, before use; late layers still attach."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.expert_offload.cpu_backend import wiring


@pytest.mark.parametrize("explicit_node", [None, 0])
def test_explicit_placement_defers_only_startup(monkeypatch, explicit_node):
    attach = Mock(return_value=object())
    monkeypatch.setattr(wiring, "maybe_attach_cpu_expert_backend", attach)
    manager = SimpleNamespace()
    kwargs = dict(routed_experts=object(), cpu_moe_config=SimpleNamespace(
        enabled=True, weight_format="gguf", host_weight_numa_node=explicit_node))
    result = wiring.attach_or_defer_cpu_expert_backend(manager, **kwargs)
    if explicit_node is None:
        assert result is attach.return_value
        attach.assert_called_once_with(**kwargs)
        return
    assert result is None
    attach.assert_not_called()
    # A repeated registration cannot duplicate storage/work.
    wiring.attach_or_defer_cpu_expert_backend(manager, **kwargs)
    assert wiring.finalize_deferred_cpu_backends(manager) == 1
    attach.assert_called_once_with(**kwargs)
    assert not manager._pending_cpu_backend_attachments
    # A later draft/MTP layer must not wait for a finalization already done.
    kwargs["routed_experts"] = object()
    assert wiring.attach_or_defer_cpu_expert_backend(manager, **kwargs) is attach.return_value
    assert attach.call_count == 2


def test_failed_cpu_load_propagates_before_graph_use(monkeypatch):
    monkeypatch.setattr(wiring, "maybe_attach_cpu_expert_backend", Mock(side_effect=RuntimeError("weight load failed")))
    manager = SimpleNamespace(_pending_cpu_backend_attachments=[{}])
    with pytest.raises(RuntimeError, match="weight load failed"):
        wiring.finalize_deferred_cpu_backends(manager)
