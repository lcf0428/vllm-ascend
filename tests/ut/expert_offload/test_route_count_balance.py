"""Token multiplicity is compute work, not repeated weight upload."""
from types import SimpleNamespace
from fractions import Fraction

import pytest

from vllm_ascend.expert_offload.host_transfer_policy import HostTransferPolicy
from vllm_ascend.expert_offload.cpu_moe_calibration import CpuMoeCosts

native=pytest.importorskip('_vllm_ascend_lrc')


def planners(cpu=100.,copy=250.,slots=2,cap=32):
    cfg=SimpleNamespace(balance_cpu_expert_us=cpu,
                        balance_copy_us=copy,balance_max_transfers=cap)
    p=HostTransferPolicy(experts=8,slots=slots,config=cfg)
    r=Fraction(str(cpu))/(Fraction(str(cpu))+Fraction(str(copy)))
    q=native.NativeHostPlanner(8,slots,0,r.numerator,r.denominator,cap,None,[0.]*8)
    return p,q


@pytest.mark.parametrize('ids,expected',[
    ([[2],[2]],[]),
    ([[2],[2],[2]],[(0,2)]),
    ([[2,3],[2,4],[2,5]],[(0,2)]),
    ([[2,3]]*6,[(0,2),(1,3)]),
])
def test_frequency_weighted_prefix(ids,expected):
    p,q=planners();mapping=[-1]*8;weights=[[1.]*len(r) for r in ids]
    a=p.plan(ids,weights,mapping);b=q.plan(ids,weights,mapping)
    assert list(a.loads)==b.loads==expected
    assert list(a.mapping)==b.mapping
    assert [list(r) for r in a.cpu_ids]==b.cpu_ids


def test_zero_weight_and_invalid_routes_are_not_compute_work():
    p,q=planners();ids=[[2,2,99,-1]];w=[[1.,0.,1.,1.]]
    assert p.plan(ids,w,[-1]*8).loads==()
    assert q.plan(ids,w,[-1]*8).loads==[]


def test_calibrated_duplicate_route_cost_changes_upload_decision():
    # A single expert with two routes costs 120 us after calibration, below
    # the 150 us upload. Two independent experts cost 200 us together.
    costs = CpuMoeCosts(first_route_ns=100_000, two_routes_ns=120_000, copy_expert_ns=150_000)
    config = SimpleNamespace(balance_max_transfers=2)
    reference = HostTransferPolicy(experts=4, slots=2, config=config, costs=costs)
    planner = native.NativeHostPlanner(
        4, 2, 0, reference.cpu_unit, reference.cpu_unit + reference.copy_unit,
        2, None, [0.0] * 4, reference.npu_fixed_unit, reference.cpu_extra_unit,
    )
    mapping = [-1] * 4
    weights = [[1.0], [1.0]]
    for ids, expected in [([[2], [2]], []), ([[2], [3]], [(0, 2)])]:
        assert list(reference.plan(ids, weights, mapping).loads) == planner.plan(ids, weights, mapping).loads == expected


def test_existing_hits_do_not_enter_missing_cpu_work():
    p,q=planners();ids=[[0,2]]*3;w=[[1.,1.]]*3;mapping=[0]+[-1]*7
    assert list(p.plan(ids,w,mapping).loads)==q.plan(ids,w,mapping).loads==[(1,2)]


def test_no_evictable_slot_or_transfer_budget_means_no_upload():
    for slots,cap in [(0,32),(2,0)]:
        p,q=planners(slots=slots,cap=cap);ids=[[2]]*8;w=[[1.]]*8
        assert list(p.plan(ids,w,[-1]*8).loads)==q.plan(ids,w,[-1]*8).loads==[]
