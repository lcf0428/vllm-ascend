"""Prefetch reads target LRC without inventing future demand observations."""

import numpy as np
import pytest

native = pytest.importorskip("_vllm_ascend_lrc")


def policy():
    return native.Policy(2, 8, 32, 0.9, 1.0, 0.5, 0.3, 0.01, False)


def planner(state):
    return native.NativeHostPlanner(8, 3, 1, 70, 520, 32, "load_balance", state, [0.0] * 8)


def test_predictions_do_not_update_any_lrc_layer():
    state = policy()
    state.observe(1, [[0, 1]], [[0.9, 0.1]])
    before = [state.snapshot(i) for i in range(2)]
    p = planner(state)
    out = p.plan_prefetch([[3, 4]], [[0.7, 0.3]], [0, 1, 2, -1, -1, -1, -1, -1], 2)
    assert len(out.loads) == 2
    assert [state.snapshot(i) for i in range(2)] == before
    p.plan([[3, 4]], [[0.7, 0.3]], out.mapping)
    assert state.step(1) == before[1]["step"] + 1
    assert state.snapshot(0) == before[0]


def test_frequency_then_router_score_global_cap_and_prediction_protection():
    p = planner(policy())
    out = p.plan_prefetch([[0, 3, 4], [3, 5, 6]], [[0.9, 0.1, 0.8], [0.1, 0.7, 0.6]], [0, 1, 2, -1, -1, -1, -1, -1], 2)
    assert [eid for _, eid in out.loads] == [3, 4]
    assert out.mapping[0] == 0
    assert out.mapping[1:3] == [-1, -1]


def test_free_slot_before_victim_and_resident_predictions_no_copy():
    p = planner(policy())
    out = p.plan_prefetch([[0, 3]], [[0.5, 0.5]], [0, -1, -1, -1, -1, -1, -1, -1], 2)
    assert out.loads == [(1, 3)]
    again = p.plan_prefetch([[0, 3]], [[0.5, 0.5]], out.mapping, 2)
    assert again.loads == []
    assert again.mapping == out.mapping


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_pinned_prepare_only_changes_mapping(dtype):
    p = planner(policy())
    ids = np.asarray([[3, 4]], dtype=dtype)
    weights = np.asarray([[0.7, 0.3]], dtype=np.float32)
    mapping = np.asarray([0, 1, 2, -1, -1, -1, -1, -1], dtype=dtype)
    ids_before, weights_before = ids.copy(), weights.copy()
    loads = p.prepare_prefetch(
        ids.ctypes.data, weights.ctypes.data, mapping.ctypes.data, 1, 2, ids.itemsize, mapping.itemsize, 2
    )
    assert len(loads) == 2
    np.testing.assert_array_equal(ids, ids_before)
    np.testing.assert_array_equal(weights, weights_before)
    assert mapping[3] >= 0 and mapping[4] >= 0


def test_invalid_predictions_are_ignored_and_bad_mapping_rejected():
    p = planner(policy())
    mapping = [0, 1, 2, -1, -1, -1, -1, -1]
    assert p.plan_prefetch([[-1, 8]], [[1.0, 1.0]], mapping, 2).loads == []
    with pytest.raises(ValueError, match="slot"):
        p.plan_prefetch([[3]], [[1.0]], [0, 0, 2, -1, -1, -1, -1, -1], 2)
    with pytest.raises(ValueError, match="cap"):
        p.plan_prefetch([[3]], [[1.0]], mapping, 0)
