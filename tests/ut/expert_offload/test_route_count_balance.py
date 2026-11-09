"""Token multiplicity is compute work, not repeated weight upload."""

from fractions import Fraction
from types import SimpleNamespace

import pytest

from vllm_ascend.expert_offload.host_transfer_policy import HostTransferPolicy

native = pytest.importorskip("_vllm_ascend_lrc")


def planners(cpu=100.0, copy=250.0, slots=2, cap=32):
    cfg = SimpleNamespace(
        transfer_policy="load_balance", balance_cpu_expert_us=cpu, balance_copy_us=copy, balance_max_transfers=cap
    )
    p = HostTransferPolicy(experts=8, slots=slots, config=cfg)
    r = Fraction(str(cpu)) / (Fraction(str(cpu)) + Fraction(str(copy)))
    q = native.NativeHostPlanner(8, slots, 0, r.numerator, r.denominator, cap, "load_balance", None, [0.0] * 8)
    return p, q


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([[2], [2]], []),
        ([[2], [2], [2]], [(0, 2)]),
        ([[2, 3], [2, 4], [2, 5]], [(0, 2)]),
        ([[2, 3]] * 6, [(0, 2), (1, 3)]),
    ],
)
def test_frequency_weighted_prefix(ids, expected):
    p, q = planners()
    mapping = [-1] * 8
    weights = [[1.0] * len(r) for r in ids]
    a = p.plan(ids, weights, mapping)
    b = q.plan(ids, weights, mapping)
    assert list(a.loads) == b.loads == expected
    assert list(a.mapping) == b.mapping
    assert [list(r) for r in a.cpu_ids] == b.cpu_ids


def test_zero_weight_and_invalid_routes_are_not_compute_work():
    p, q = planners()
    ids = [[2, 2, 99, -1]]
    w = [[1.0, 0.0, 1.0, 1.0]]
    assert p.plan(ids, w, [-1] * 8).loads == ()
    assert q.plan(ids, w, [-1] * 8).loads == []


def test_existing_hits_do_not_enter_missing_cpu_work():
    p, q = planners()
    ids = [[0, 2]] * 3
    w = [[1.0, 1.0]] * 3
    mapping = [0] + [-1] * 7
    assert list(p.plan(ids, w, mapping).loads) == q.plan(ids, w, mapping).loads == [(1, 2)]


def test_no_evictable_slot_or_transfer_budget_means_no_upload():
    for slots, cap in [(0, 32), (2, 0)]:
        p, q = planners(slots=slots, cap=cap)
        ids = [[2]] * 8
        w = [[1.0]] * 8
        assert list(p.plan(ids, w, [-1] * 8).loads) == q.plan(ids, w, [-1] * 8).loads == []
