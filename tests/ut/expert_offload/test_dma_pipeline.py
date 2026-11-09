from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.expert_offload.h2d_transfer import H2DCopyTask, TorchCopyH2DTransport


def make_manager(events, fail=False):
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._prefetch_trace = MagicMock()
    manager._get_h2d_transport = lambda: TorchCopyH2DTransport()

    def build(layer, layer_idx, eid, slot):
        events.append(('build', eid))
        def copy(source, non_blocking):
            assert non_blocking
            events.append(('copy', eid))
            if fail:
                raise RuntimeError('DMA submit failed')
        return [H2DCopyTask(object(), SimpleNamespace(copy_=copy), 16)]

    manager._build_expert_h2d_tasks = build
    manager._refresh_expert_fp32_scale = lambda layer, slot: events.append(('scale', slot))
    return manager


@pytest.mark.parametrize('timed', [False, True])
def test_first_expert_submitted_before_next_build_and_scales_last(timed):
    events = []
    manager = make_manager(events)
    timing = {} if timed else None
    manager._load_expert_weights_into_slots(object(), 0, [(3, 9), (4, 8)], timing)
    assert events == [('build', 9), ('copy', 9), ('build', 8), ('copy', 8), ('scale', 3), ('scale', 4)]
    if timed:
        first, second = timing['expert_submissions']
        assert first['first_copy_call_return_ns'] <= second['descriptor_begin_ns']
        assert timing['first_copy_call_enter_ns'] == first['first_copy_call_enter_ns']
        assert timing['descriptor_end_ns'] <= timing['first_copy_call_enter_ns']
        assert timing['batch_copy_return_ns'] >= second['batch_copy_return_ns']


def test_submit_failure_stops_later_experts_and_scale_updates():
    events = []
    manager = make_manager(events, fail=True)
    with pytest.raises(RuntimeError, match='DMA submit failed'):
        manager._load_expert_weights_into_slots(object(), 0, [(3, 9), (4, 8)])
    assert events == [('build', 9), ('copy', 9)]


def test_empty_loads_do_not_submit_copy():
    events = []
    make_manager(events)._load_expert_weights_into_slots(object(), 0, [])
    assert not events
