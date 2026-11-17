import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.expert_offload.cpu_backend.wiring as wiring


@pytest.fixture
def config_classes():
    # Exercise the production classes without importing the full worker dependency graph.
    path = Path(wiring.__file__).parents[2] / "ascend_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("CpuMoeConfig", "ExpertOffloadConfig")]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def cpu_config_class(config_classes):
    return config_classes["CpuMoeConfig"]


@pytest.fixture
def offload_config_class(config_classes, monkeypatch):
    predictors = SimpleNamespace(valid_predictor_names=lambda: ["fate", "mode2_har", "mode2_prevhfr"])
    monkeypatch.setitem(sys.modules, "vllm_ascend.expert_offload.expert_predictor", predictors)
    return config_classes["ExpertOffloadConfig"]


@pytest.mark.parametrize("predictor", ["mode2_har", "mode2_prevhfr"])
def test_cpu_moe_rejects_trained_prefetch_at_startup(offload_config_class, tmp_path, predictor):
    checkpoint = tmp_path / "predictor.pt"
    checkpoint.touch()
    config = {
        "expert_offload": True,
        "expert_prefetch_enabled": True,
        "expert_predictor": predictor,
        "expert_predictor_ckpt": str(checkpoint),
    }
    paging = offload_config_class(config)
    assert not paging.cpu_moe.enabled
    assert paging.expert_predictor == predictor
    with pytest.raises(ValueError, match=f"CPU MoE prefetch only supports expert_predictor='fate'; got '{predictor}'"):
        offload_config_class({**config, "cpu_moe": {"enabled": True}})


@pytest.mark.parametrize("prefetch_enabled", [False, True])
def test_cpu_moe_accepts_default_fate(offload_config_class, prefetch_enabled):
    config = offload_config_class({
        "expert_offload": True,
        "expert_prefetch_enabled": prefetch_enabled,
        "cpu_moe": {"enabled": True},
    })
    assert config.expert_predictor == "fate"


def test_cpu_config_disabled_by_default(cpu_config_class):
    cfg = cpu_config_class()
    assert not cfg.enabled
    assert cfg.cpuinfer_threads is None
    assert cfg.numa_nodes is None


@pytest.mark.parametrize(("key", "value"), [
    ("backend", "kt_kernel"),
    ("layers", [999]),
    ("weight_path", "/unused/{layer_idx}.gguf"),
    ("weight_format", "gguf"),
    ("native_dispatch", False),
    ("transfer_policy", "load_balance"),
    ("transfer_scheduler", "cpu"),
])
def test_cpu_moe_rejects_retired_options(cpu_config_class, key, value):
    config = {"enabled": True}
    assert cpu_config_class(config).enabled
    with pytest.raises(ValueError, match=f"Config has no attribute '{key}'"):
        cpu_config_class({**config, key: value})


@pytest.mark.parametrize("capacity", [0, -1, True, "6"])
def test_cpu_config_rejects_invalid_capacity(cpu_config_class, capacity):
    with pytest.raises((TypeError, ValueError)):
        cpu_config_class({"max_num_tokens": capacity})


def test_cpu_config_rejects_empty_threadpool(cpu_config_class):
    with pytest.raises(ValueError, match="cpuinfer_threads must be >= cpu_moe.threadpool_count"):
        cpu_config_class({"enabled": True, "cpuinfer_threads": 1, "threadpool_count": 2})


@pytest.mark.parametrize("capacity", [10, 12, 16])
def test_wiring_checks_verify_capacity_before_backend_creation(cpu_config_class, monkeypatch, capacity):
    config_module = types.ModuleType("vllm.config")
    config_module.get_current_vllm_config = lambda: SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=2),
    )
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setattr(wiring, "model_layer_index", lambda _: 3)
    created = []

    def factory(config, *, native_weights):
        created.append(config)
        return SimpleNamespace(max_num_tokens=config.max_num_tokens, submit=lambda **_: None)

    monkeypatch.setattr(wiring, "KtKernelCPUExpertBackend", factory)
    cfg = cpu_config_class(
        {"enabled": True, "max_num_tokens": capacity}
    )
    cfg.startup_affinity = SimpleNamespace(
        compute_cpus=((3, 7),), numa_nodes=(2,), poll_cpu=10, completion_cpu=11,
    )
    args = dict(
        routed_experts=torch.nn.Module(),
        layer_name="model.layers.3.mlp",
        moe_config=SimpleNamespace(
            ep_size=1, num_logical_experts=4, experts_per_token=2, hidden_dim=4, intermediate_size_per_partition=8
        ),
        cpu_moe_config=cfg,
        enable_multi_card=False,
        tp_size=1,
    )
    assert wiring.maybe_attach_cpu_expert_backend(**args, native_weights=object()).max_num_tokens == capacity
    assert len(created) == 1
    assert (created[0].poll_cpu, created[0].completion_cpu) == (10, 11)
    assert created[0].cpuinfer_threads == 2
    assert created[0].compute_cpus == ((3, 7),)
    assert created[0].numa_nodes == (2,)


@pytest.mark.parametrize("tp, ep, multi_card", [(2, 1, False), (1, 2, False), (1, 1, True)])
def test_wiring_retains_single_card_constraint(cpu_config_class, monkeypatch, tp, ep, multi_card):
    monkeypatch.setattr(wiring, "model_layer_index", lambda _: 3)
    cfg = cpu_config_class({"enabled": True})
    with pytest.raises(ValueError, match="single-card"):
        wiring.maybe_attach_cpu_expert_backend(
            routed_experts=torch.nn.Module(),
            layer_name="model.layers.3.mlp",
            moe_config=SimpleNamespace(ep_size=ep),
            cpu_moe_config=cfg,
            enable_multi_card=multi_card,
            tp_size=tp,
        )
