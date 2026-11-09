"""Allocation affinity is temporary even if the driver rejects an allocation."""

from unittest.mock import Mock

import pytest

from vllm_ascend.expert_offload import host_weight_allocator as module


def test_default_does_not_change_affinity(monkeypatch):
    change = Mock()
    monkeypatch.setattr(module.os, "sched_setaffinity", change)
    allocate = Mock(return_value=object())
    result = module.HostWeightAllocator().allocate(allocate, (2, 4), "dtype")
    assert result is allocate.return_value
    allocate.assert_called_once_with((2, 4), "dtype")
    change.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_selected_node_restores_caller_affinity(monkeypatch, failure):
    monkeypatch.setattr(module.Path, "read_text", lambda _: "0-31,64-95\n")
    current = {32, 33, 62, 63}
    changes = []
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: set(current))

    def change(_, mask):
        changes.append(set(mask))
        current.clear()
        current.update(mask)

    monkeypatch.setattr(module.os, "sched_setaffinity", change)

    def allocate(shape, dtype):
        assert current == {0}
        if failure:
            raise RuntimeError("driver allocation failed")
        return shape, dtype

    monkeypatch.setattr(module.HostWeightAllocator, "_save_policy", lambda _: "old_strict_policy")
    policy_events = []
    monkeypatch.setattr(module.HostWeightAllocator, "_select_policy", lambda _: policy_events.append("select"))
    monkeypatch.setattr(module.HostWeightAllocator, "_restore_policy", lambda _, p: policy_events.append(p))
    allocator = module.HostWeightAllocator(0)
    if failure:
        with pytest.raises(RuntimeError, match="driver allocation failed"):
            allocator.allocate(allocate, (4,), "dtype")
    else:
        assert allocator.allocate(allocate, (4,), "dtype") == ((4,), "dtype")
    assert current == {32, 33, 62, 63}
    assert changes == [{0}, {32, 33, 62, 63}]
    assert policy_events == ["select", "old_strict_policy"]


@pytest.mark.parametrize("node", [-1, True, "0", 0.5])
def test_invalid_node_is_rejected(node):
    with pytest.raises(ValueError, match="host_weight_numa_node"):
        module.HostWeightAllocator(node)
