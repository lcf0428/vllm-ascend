import json
import threading
import time

from vllm_ascend.expert_offload.dma_submit_timing import DmaSubmitTrace,stamp
from vllm_ascend.expert_offload.h2d_transfer import TorchCopyH2DTransport,H2DCopyTask


def test_inactive_trace_records_nothing(tmp_path):
    trace=DmaSubmitTrace()
    assert trace.begin({'event':'unused'}) is None
    stamp(None,'unused')
    trace.end(None)
    trace.set_capture(False,tmp_path)
    assert not list(tmp_path.iterdir())


def test_profile_flush_is_buffered_and_sessions_append(tmp_path):
    trace=DmaSubmitTrace()
    for i in range(2):
        trace.set_capture(True,tmp_path)
        record=trace.begin({'event':'test','i':i})
        stamp(record,'begin');stamp(record,'end')
        assert record['end_ns']>=record['begin_ns']
        assert record['end_cpu_ns']>=record['begin_cpu_ns']
        trace.end(record)
        trace.set_capture(False,tmp_path)
    f,=list(tmp_path.glob('cpu_dma_submit_*.jsonl'))
    assert [json.loads(l)['i'] for l in f.open()]==[0,1]


def test_timing_keeps_copy_order_and_first_copy_boundaries():
    order=[]
    class Destination:
        def copy_(self,source,non_blocking):
            order.append((source,non_blocking))
    tasks=[H2DCopyTask(i,Destination(),1) for i in range(3)]
    trace={};transport=TorchCopyH2DTransport()
    transport.copy_batch_timed(tasks,trace)
    assert order==[(0,True),(1,True),(2,True)]
    assert trace['first_copy_call_return_ns']>=trace['first_copy_call_enter_ns']
    assert trace['first_copy_call_return_cpu_ns']>=trace['first_copy_call_enter_cpu_ns']
