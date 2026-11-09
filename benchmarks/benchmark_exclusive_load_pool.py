"""CPU-only transpose-copy benchmark; includes staging and final drain.

Run in a PyTorch environment. Does not measure checkpoint I/O or NZ conversion.
"""

import argparse
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size-mb", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(value <= 0 for value in vars(args).values()):
        parser.error("All sizes, counts and repeat values must be positive")

    # Avoid importing the vllm_ascend package so the benchmark also runs on
    # CPU-only PyTorch installations without torch_npu/vLLM.
    path = Path(__file__).resolve().parents[1] / "vllm_ascend/expert_offload/load_buffer_pool.py"
    spec = importlib.util.spec_from_file_location("load_buffer_pool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = torch.full((args.rows, args.cols), 17, dtype=torch.uint8)
    destinations = [torch.empty((args.cols, 2 * args.rows), dtype=torch.uint8) for _ in range(args.experts)]
    nbytes = source.numel() * source.element_size()
    original_threads = torch.get_num_threads()
    for repeat in range(args.repeats):
        # Alternate order to reduce warm-cache/order bias.
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            for destination in destinations:
                destination.zero_()
            pool = executor = None
            started = time.perf_counter()
            try:
                if enabled:
                    torch.set_num_threads(1)
                    executor = ThreadPoolExecutor(max_workers=args.workers)
                    pool = module.LoadBufferPool(
                        args.pool_size_mb * 1024 * 1024, executor,
                        lambda size: torch.empty(size, dtype=torch.uint8),
                    )
                for destination in destinations:
                    for offset in (0, args.rows):
                        target = destination[:, offset:offset + args.rows]
                        if pool is not None and pool.submit(
                            nbytes,
                            lambda storage, start: storage.narrow(0, start, nbytes).reshape(source.shape),
                            lambda view: view.copy_(source),
                            lambda view, dst=target: dst.copy_(view.t()),
                        ):
                            continue
                        target.copy_(source.t())
            finally:
                try:
                    if pool is not None:
                        pool.close()
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True)
                    torch.set_num_threads(original_threads)
            elapsed = time.perf_counter() - started
            for destination in destinations:
                assert bool((destination == 17).all()), "Incorrect transpose copy"
            print(json.dumps({
                "repeat": repeat, "mode": "pooled" if enabled else "synchronous",
                "seconds_including_drain": elapsed,
                "destination_gib_per_second": 2 * args.experts * nbytes / (1024 ** 3) / elapsed,
                "staging_capacity_bytes": pool.capacity_bytes if pool else 0,
                "peak_leased_bytes": pool.peak_in_use_bytes if pool else 0,
                "synchronous_fallbacks": pool.synchronous if pool else 0,
                "sync_torch_threads": original_threads, "pool_workers": args.workers if enabled else 0,
            }), flush=True)


if __name__ == "__main__":
    main()
