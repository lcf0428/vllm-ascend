"""Real tensor regression for bounded checkpoint staging into pinned Host slots."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager


@pytest.mark.parametrize("enabled", [False, True])
def test_parallel_checkpoint_shards_match_synchronous_layout(enabled):
    torch.npu.set_device(0)
    original_threads = torch.get_num_threads()
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = SimpleNamespace(
        exclusive_expert_storage=True, exclusive_load_pool_enabled=enabled,
        exclusive_load_pool_size_mb=1, shard_per_rank=False,
    )
    manager._native_shared_weights = False
    manager._load_pool = manager._exclusive_load_pool = None
    manager._saved_num_threads = None
    manager._load_phase_start = manager._weight_load_secs = 0.0
    manager._load_futures = []
    manager._weight_load_calls = 0
    w13 = torch.zeros((128, 256), dtype=torch.uint8, pin_memory=True)
    w2 = torch.zeros((128, 128), dtype=torch.uint8, pin_memory=True)
    scales = torch.zeros(256, dtype=torch.float32, pin_memory=True)
    manager.w13_weights_cpu = [[None, w13]]
    manager.w2_weights_cpu = [[None, w2]]
    manager.scale_cpu_buffers = {"w13_weight_scale": [[None, scales]]}
    try:
        for shard, value in (("w3", 3), ("w1", 1)):
            source = torch.full((128, 128), value, dtype=torch.uint8)
            manager.load_w13(0, 1, source, shard)
            source.fill_(255)
            scale = torch.full((128,), float(value))
            manager._load_scale_shard(0, 1, "w13_weight_scale", shard, scale)
            scale.fill_(255)
        source = torch.arange(128 * 128, dtype=torch.int32).to(torch.uint8).reshape(128, 128)
        expected_w2 = source.t().clone()
        manager.load_w2(0, 1, source)
        source.fill_(255)
    finally:
        manager.drain_load_pool()
    assert torch.get_num_threads() == original_threads
    assert manager._exclusive_load_pool is None
    assert manager._load_pool is None
    torch.testing.assert_close(w13[:, :128], torch.full_like(w13[:, :128], 1))
    torch.testing.assert_close(w13[:, 128:], torch.full_like(w13[:, 128:], 3))
    torch.testing.assert_close(w2, expected_w2)
    torch.testing.assert_close(scales, torch.cat((torch.ones(128), torch.full((128,), 3.0))))
