from types import SimpleNamespace
import random
import pytest
from vllm_ascend.expert_offload.host_transfer_policy import HostTransferPolicy
native=pytest.importorskip('_vllm_ascend_lrc')


def plan_pair(ids, fixed, cpu=100., copy=250.):
    cfg=SimpleNamespace(transfer_policy='load_balance',balance_cpu_expert_us=cpu,
                        balance_copy_us=copy,balance_npu_fixed_us=fixed,balance_max_transfers=32)
    p=HostTransferPolicy(experts=8,slots=2,config=cfg)
    q=native.NativeHostPlanner(8,2,0,p.cpu_unit,p.cpu_unit+p.copy_unit,32,
                              'load_balance',None,[0.]*8,p.npu_fixed_unit)
    weights=[[1.]*len(row) for row in ids];mapping=[-1]*8
    a=p.plan(ids,weights,mapping);b=q.plan(ids,weights,mapping)
    assert list(a.loads)==b.loads
    assert list(a.mapping)==b.mapping
    assert [list(r) for r in a.cpu_ids]==b.cpu_ids
    return list(a.loads)


def test_fixed_cost_can_keep_a_small_job_on_cpu():
    assert plan_pair([[2]]*3,0)==[(0,2)]
    assert plan_pair([[2]]*3,200)==[]


def test_fixed_cost_is_once_per_layer_and_reduces_borderline_uploads():
    assert plan_pair([[2,3]]*6,0)==[(0,2),(1,3)]
    assert plan_pair([[2,3]]*6,200)==[(0,2)]
    # Enough CPU work still justifies one upload: 500 versus 250+200.
    assert plan_pair([[2]]*5,200)==[(0,2)]


def test_decimal_costs_and_integer_ratio_scaling_agree():
    assert plan_pair([[2]]*5,200,cpu=93.16,copy=275.828)==[]
    assert plan_pair([[2]]*6,200,cpu=93.16,copy=275.828)==[(0,2)]
    assert plan_pair([[2]]*10,200,cpu=3,copy=6)==[]


def test_python_cpp_parity_with_repeated_routes():
    rng=random.Random(200)
    for _ in range(40):
        ids=[[rng.randrange(8) for _ in range(6)] for _ in range(rng.randrange(1,9))]
        plan_pair(ids,200,cpu=93.16,copy=275.828)
