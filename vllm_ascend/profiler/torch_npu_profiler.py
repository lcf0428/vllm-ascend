#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from contextlib import suppress
from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler

from vllm_ascend.ascend_config import get_ascend_config


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)
        self._cpu_moe_anchor_directory = profiler_config.torch_profiler_dir
        self.profiler: Any = self._create_profiler(profiler_config, trace_name)

    @staticmethod
    def _create_profiler(profiler_config: ProfilerConfig, trace_name: str) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        msmonitor_use_daemon = False
        with suppress(RuntimeError):
            msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
        if msmonitor_use_daemon:
            raise RuntimeError(
                "additional_config.msmonitor_use_daemon and torch profiler cannot be enabled at the same time."
            )

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
            # The with_stack option in torch_npu.profiler introduces significant time overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                worker_name=trace_name,
            ),
        )

    @staticmethod
    def _set_cpu_moe_capture(enabled: bool, directory=None) -> None:
        # Optional kt backend: enable detailed native events only in a capture.
        with suppress(ImportError):
            from kt_kernel import kt_kernel_ext
            setter = getattr(kt_kernel_ext, "set_lifecycle_enabled", None)
            if setter is not None:
                setter(enabled)
        with suppress(ImportError):
            from vllm_ascend.expert_offload.cpu_backend import poll_submission
            worker = poll_submission._WORKER
            if worker is not None and hasattr(worker, 'set_dma_capture'):
                worker.set_dma_capture(enabled, directory)

    def _start(self) -> None:
        import os
        import torch
        if os.environ.get("CPU_MOE_CLOCK_ANCHORS") == "1":
            torch.npu.synchronize()
        self.profiler.start()
        if os.environ.get("CPU_MOE_CLOCK_ANCHORS") == "1":
            from vllm_ascend.expert_offload.cpu_backend.clock_anchor_helper import record_clock_anchor
            record_clock_anchor("start", self._cpu_moe_anchor_directory)
        self._set_cpu_moe_capture(True, self._cpu_moe_anchor_directory)

    def _stop(self) -> None:
        import os
        import torch
        if os.environ.get("CPU_MOE_CLOCK_ANCHORS") == "1":
            # Caller must finish benchmark responses before requesting stop.
            # Drain queued NPU work, including CPU callback result transfers.
            torch.npu.synchronize()
            from vllm_ascend.expert_offload.cpu_backend.clock_anchor_helper import record_clock_anchor
            record_clock_anchor("stop", self._cpu_moe_anchor_directory)
        self._set_cpu_moe_capture(False, self._cpu_moe_anchor_directory)
        self.profiler.stop()

    def _profiler_step(self) -> bool:
        return True
