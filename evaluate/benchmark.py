"""Inference speed benchmark for Korean NER."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = REPO_ROOT / "weights" / "best_model.pt"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 192
OUTPUT_DIR = REPO_ROOT / "runs" / "benchmark"
DEFAULT_WARMUP = 10
DEFAULT_RUNS = 50


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Benchmark Korean NER inference latency.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to a raw or wrapped model checkpoint.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name or path.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Aligned token sequence length.")
    parser.add_argument("--device", default="cpu", help="Torch device to benchmark. Defaults to explicit CPU.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup passes per sample before timing.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Measured passes per sample.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for benchmark_results.json.")
    return parser.parse_args(argv)


def percentile(values: Sequence[float], percent: float) -> float:
    """Compute a linear percentile without requiring numpy."""
    if not values:
        raise ValueError("percentile requires at least one value")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percent / 100)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def benchmark_inference(model, prepared_inputs, device, torch_module, *, num_warmup: int, num_runs: int):
    """Benchmark prepared aligned inputs and report the sample size explicitly."""
    if num_warmup < 0 or num_runs <= 0:
        raise ValueError("num_warmup must be >= 0 and num_runs must be > 0")

    model.eval()
    print(f"  Warmup: {num_warmup} passes per sample")
    for _ in range(num_warmup):
        for inp in prepared_inputs:
            with torch_module.no_grad():
                _ = model(inp["input_ids"], inp["attention_mask"])

    if device.type == "cuda":
        torch_module.cuda.synchronize()

    print(f"  Measuring: {num_runs} passes per sample across {len(prepared_inputs)} samples")
    latencies = []
    for _ in range(num_runs):
        for inp in prepared_inputs:
            if device.type == "cuda":
                torch_module.cuda.synchronize()
            start = time.perf_counter()
            with torch_module.no_grad():
                _ = model(inp["input_ids"], inp["attention_mask"])
            if device.type == "cuda":
                torch_module.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    mean_latency = mean(latencies)
    return {
        "device": str(device),
        "warmup_passes_per_sample": num_warmup,
        "measured_passes_per_sample": num_runs,
        "text_sample_count": len(prepared_inputs),
        "timed_forward_passes": len(latencies),
        "mean_latency_ms": mean_latency,
        "std_latency_ms": pstdev(latencies) if len(latencies) > 1 else 0.0,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "throughput_samples_per_sec": 1000 / mean_latency,
    }


def prepare_inputs(texts, tokenizer, label_maps, max_length, device, torch_module):
    """Align original text, preserving spaces, and create model tensors."""
    from korean_ner.alignment import align_text

    prepared = []
    for text in texts:
        encoding = align_text(
            text,
            tokenizer,
            max_length=max_length,
            label_to_id=label_maps.label_to_id,
            id_to_label=label_maps.id_to_label,
        )
        prepared.append(
            {
                "input_ids": torch_module.tensor([encoding.input_ids], dtype=torch_module.long, device=device),
                "attention_mask": torch_module.tensor([encoding.attention_mask], dtype=torch_module.long, device=device),
                "text_length": len(text),
            }
        )
    return prepared


def test_texts() -> list[str]:
    return [
        "삼성전자",
        "삼성전자 이재용 회장이 서울에서 회의를 열었다.",
        "삼성전자 이재용 회장이 2024년 1월 15일 서울에서 새 제품을 발표했다.",
        "한국은행은 오늘 오전 서울에서 올해 경제 전망을 발표하고 기준금리를 논의했다. 회의에는 전문가 약 500명이 참석했다.",
    ]


def main(argv: Sequence[str] | None = None):
    import torch
    from transformers import AutoTokenizer

    from korean_ner.checkpoint import load_model_state
    from korean_ner.labels import build_label_maps
    from korean_ner.model import KoElectraNER

    args = parse_args(argv)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Korean NER Inference Speed Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Warmup: {args.warmup} passes per sample")
    print(f"Measured runs: {args.runs} passes per sample")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    label_maps = build_label_maps()
    texts = test_texts()
    prepared_inputs = prepare_inputs(texts, tokenizer, label_maps, args.max_length, device, torch)

    model = KoElectraNER(args.model_name, len(label_maps.names))
    load_model_state(model, args.model_path, map_location=device)
    model.to(device)

    result = benchmark_inference(
        model,
        prepared_inputs,
        device,
        torch,
        num_warmup=args.warmup,
        num_runs=args.runs,
    )
    result["model_name"] = args.model_name
    result["max_length"] = args.max_length
    result["text_lengths"] = [len(text) for text in texts]
    result["device_name"] = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU ({torch.get_num_threads()} threads)"
    )

    print("\nBenchmark Summary")
    print("-" * 40)
    print(f"Samples: {result['text_sample_count']}")
    print(f"Timed forward passes: {result['timed_forward_passes']}")
    print(f"Latency: {result['mean_latency_ms']:.2f} +/- {result['std_latency_ms']:.2f} ms")
    print(f"Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")

    output_path = output_dir / "benchmark_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
