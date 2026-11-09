"""Native admission must preserve ownership, migration decisions and LRC state."""
import random
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.expert_offload.host_transfer_policy import HostTransferPolicy
from vllm_ascend.expert_offload.lrc_policy import CppLRCExpertCachePolicy

native = pytest.importorskip("_vllm_ascend_lrc")


@pytest.mark.parametrize("policy", ["first_miss", "frequency", "load_balance"])
@pytest.mark.parametrize("cache", [False, True])
def test_repeated_native_plans_match_reference(policy, cache):
    rng = random.Random(817)
    config = SimpleNamespace(transfer_policy=policy, balance_cpu_expert_us=70.0,
                             balance_copy_us=450.0, balance_max_transfers=32)
    states = [CppLRCExpertCachePolicy(num_layers=1, num_experts=32, cache_size=8, topk=6)
              if cache else None for _ in range(2)]
    if cache:
        for state in states:
            state.seed_layer_hotness(0, {0: 10, 3: 5, 7: 2})
    reference = HostTransferPolicy(experts=32, slots=8, config=config, lrc=states[0])
    ratio = Fraction(70, 520)
    planner = native.NativeHostPlanner(32, 8, 0, ratio.numerator, ratio.denominator, 32,
                                      policy, states[1]._native if cache else None, [0.0] * 32)
    mapping = list(range(8)) + [-1] * 24
    for _ in range(200):
        rows = rng.randint(1, 4)
        ids = [[rng.randrange(-1, 34) for _ in range(6)] for _ in range(rows)]
        weights = [[rng.choice([0.0, 0.1, 0.25, 0.5]) for _ in row] for row in ids]
        expected = reference.plan(ids, weights, mapping)
        actual = planner.plan(ids, weights, mapping)
        assert actual.mapping == list(expected.mapping)
        assert actual.loads == list(expected.loads)
        assert actual.cpu_ids == [list(row) for row in expected.cpu_ids]
        assert actual.cpu_weights == [list(row) for row in expected.cpu_weights]
        mapping = actual.mapping
        if cache:
            assert states[0]._native.snapshot(0) == states[1]._native.snapshot(0)


@pytest.mark.parametrize("id_type", [np.int32, np.int64])
@pytest.mark.parametrize("map_type", [np.int32, np.int64])
def test_native_prepare_mutates_only_owned_pinned_views(id_type, map_type):
    planner = native.NativeHostPlanner(8, 2, 0, 1, 2, 2, "load_balance", None, [0.0] * 8)
    ids = np.asarray([[0, 2, 3, 4]], dtype=id_type)
    weights = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    mapping = np.asarray([0, 1] + [-1] * 6, dtype=map_type)
    loads = planner.prepare(ids.ctypes.data, weights.ctypes.data, mapping.ctypes.data,
                            1, 4, ids.itemsize, mapping.itemsize)
    assert loads == [(1, 2)]
    assert ids.tolist() == [[-1, -1, 3, 4]]
    np.testing.assert_array_equal(weights, np.asarray([[0, 0, 0.3, 0.4]], np.float32))
    assert mapping.tolist() == [0, -1, 1, -1, -1, -1, -1, -1]


def test_invalid_mapping_and_buffers_fail_before_submission():
    planner = native.NativeHostPlanner(4, 2, 0, 1, 2, 2, "load_balance", None, [0.0] * 4)
    with pytest.raises(ValueError, match="slot"):
        planner.plan([[0]], [[1.0]], [0, 0, -1, -1])
    with pytest.raises(ValueError, match="buffers"):
        planner.prepare(0, 0, 0, 1, 1, 8, 4)
