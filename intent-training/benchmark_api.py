"""Benchmark the FastAPI intent endpoint across concurrency levels.

Example:
    /mnt/HDD8TB/aman_ws/stt/.venv/bin/python3 benchmark_api.py \
        --audio /path/to/sample.wav --concurrencies 1,2,4,8 --requests 40
"""

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from pathlib import Path

import httpx


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def make_request_kwargs(audio_name, audio_bytes, top_k):
    data = {}
    if top_k is not None:
        data["top_k"] = str(top_k)
    return {
        "data": data,
        "files": {"audio": (audio_name, audio_bytes, "application/octet-stream")},
    }


async def submit_request(client, url, audio_name, audio_bytes, top_k):
    started = time.perf_counter()
    try:
        response = await client.post(
            url, **make_request_kwargs(audio_name, audio_bytes, top_k)
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "latency_ms": latency_ms,
            "status_code": response.status_code,
            "error": None if response.is_success else response.text[:300],
        }
    except httpx.HTTPError as exc:
        return {
            "latency_ms": (time.perf_counter() - started) * 1000,
            "status_code": None,
            "error": str(exc),
        }


async def run_level(client, url, audio_name, audio_bytes, top_k, concurrency, request_count):
    next_request = 0
    next_request_lock = asyncio.Lock()
    results = []

    async def worker():
        nonlocal next_request
        while True:
            async with next_request_lock:
                if next_request >= request_count:
                    return
                next_request += 1
            results.append(
                await submit_request(client, url, audio_name, audio_bytes, top_k)
            )

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed_s = time.perf_counter() - started
    return results, elapsed_s


def summarize(concurrency, results, elapsed_s):
    successful = [
        item["latency_ms"] for item in results
        if item["status_code"] is not None and 200 <= item["status_code"] < 300
    ]
    status_codes = Counter(
        str(item["status_code"]) if item["status_code"] is not None else "request_error"
        for item in results
    )
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "elapsed_seconds": round(elapsed_s, 3),
        "throughput_rps": round(len(successful) / elapsed_s, 3) if elapsed_s else 0.0,
        "latency_ms": {
            "min": round(min(successful), 2) if successful else None,
            "mean": round(sum(successful) / len(successful), 2) if successful else None,
            "p50": round(percentile(successful, 0.50), 2) if successful else None,
            "p95": round(percentile(successful, 0.95), 2) if successful else None,
            "p99": round(percentile(successful, 0.99), 2) if successful else None,
            "max": round(max(successful), 2) if successful else None,
        },
        "status_codes": dict(status_codes),
    }


def print_summary(summary):
    latency = summary["latency_ms"]
    print(
        f"concurrency={summary['concurrency']:>2}  "
        f"success={summary['successful']}/{summary['requests']}  "
        f"throughput={summary['throughput_rps']:.2f} req/s  "
        f"p50={latency['p50']} ms  p95={latency['p95']} ms  "
        f"p99={latency['p99']} ms  max={latency['max']} ms  "
        f"statuses={summary['status_codes']}"
    )


async def main_async(args):
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audio_bytes = audio_path.read_bytes()
    if not audio_bytes:
        raise ValueError("Audio file is empty")

    concurrencies = [int(value) for value in args.concurrencies.split(",") if value]
    if not concurrencies or any(value < 1 for value in concurrencies):
        raise ValueError("--concurrencies must contain positive integers")

    timeout = httpx.Timeout(args.timeout_seconds)
    limits = httpx.Limits(max_connections=max(concurrencies), max_keepalive_connections=max(concurrencies))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health = await client.get(args.health_url)
        health.raise_for_status()
        print(f"Server health: {health.json()}")

        if args.warmup:
            print(f"Warming up with {args.warmup} request(s)...")
            for _ in range(args.warmup):
                result = await submit_request(
                    client, args.url, audio_path.name, audio_bytes, args.top_k
                )
                if result["error"]:
                    raise RuntimeError(f"Warmup failed: {result['error']}")

        summaries = []
        for concurrency in concurrencies:
            results, elapsed_s = await run_level(
                client, args.url, audio_path.name, audio_bytes, args.top_k,
                concurrency, args.requests,
            )
            summary = summarize(concurrency, results, elapsed_s)
            summaries.append(summary)
            print_summary(summary)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
        print(f"Saved benchmark JSON -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Audio file sent for every request")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/predict")
    parser.add_argument("--health-url", default="http://127.0.0.1:8000/health")
    parser.add_argument("--concurrencies", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=40,
                        help="Total measured requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--top-k", type=int,
                        help="Optional: include top_k in every API request")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--json-out", help="Optional path for machine-readable results")
    args = parser.parse_args()
    if args.requests < 1 or args.warmup < 0:
        parser.error("--requests must be positive and --warmup cannot be negative")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
