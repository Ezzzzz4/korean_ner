"""
Inference Speed Benchmark for Korean NER
Measures throughput, latency, and GPU/CPU comparison.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
import time
import numpy as np
from tqdm import tqdm
import json
import os

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "../weights/best_model.pt"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 128
OUTPUT_DIR = "../assets"

# KLUE NER Labels
LABEL_LIST = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']
NUM_LABELS = len(LABEL_LIST)


# ============================================================================
# Model Definition
# ============================================================================
class KoElectraNER(nn.Module):
    def __init__(self, model_name, num_labels, num_samples=5):
        super().__init__()
        self.num_labels = num_labels
        self.num_samples = num_samples
        
        self.electra = AutoModel.from_pretrained(model_name)
        hidden = self.electra.config.hidden_size
        
        self.lstm = nn.LSTM(hidden, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None):
        enc = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = enc.last_hidden_state
        lstm_out, _ = self.lstm(seq_out)
        emissions = self.classifier(self.dropout(lstm_out))
        mask = attention_mask.bool()
        
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction='mean')
            return loss, emissions
        return self.crf.decode(emissions, mask=mask)


def benchmark_inference(model, tokenizer, texts, device, batch_size=1, num_warmup=10, num_runs=100):
    """Benchmark inference speed."""
    model.eval()
    
    # Prepare all inputs
    all_inputs = []
    for text in texts:
        chars = list(text)
        inputs = tokenizer(
            chars,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding='max_length'
        )
        all_inputs.append({
            'input_ids': inputs['input_ids'].to(device),
            'attention_mask': inputs['attention_mask'].to(device)
        })
    
    # Warmup
    print(f"  Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        for inp in all_inputs:
            with torch.no_grad():
                _ = model(inp['input_ids'], inp['attention_mask'])
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    latencies = []
    
    for _ in tqdm(range(num_runs), desc="  Running"):
        for inp in all_inputs:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(inp['input_ids'], inp['attention_mask'])
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_samples_per_sec': 1000 / np.mean(latencies),
        'num_samples': len(latencies)
    }


def main():
    print("=" * 60)
    print("Korean NER Inference Speed Benchmark")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Test sentences of varying lengths
    test_texts = [
        "삼성전자",  # Short (4 chars)
        "삼성전자 이재용 회장이 서울에서",  # Medium (18 chars)
        "삼성전자 이재용 회장이 2024년 1월 15일 서울에서 기자회견을 열었다.",  # Long (42 chars)
        "대한민국 서울특별시 강남구에서 열린 행사에 약 500명이 참석했습니다. 네이버와 카카오 관계자도 참석했습니다.",  # Very long (67 chars)
    ]
    
    results = {}
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # ========== GPU Benchmark ==========
    if torch.cuda.is_available():
        print("\n" + "=" * 40)
        print("GPU Benchmark (CUDA)")
        print("=" * 40)
        
        device = torch.device('cuda')
        print(f"Device: {torch.cuda.get_device_name(0)}")
        
        # Load model
        model = KoElectraNER(MODEL_NAME, NUM_LABELS)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        # Benchmark different text lengths
        gpu_results = {}
        for i, text in enumerate(test_texts):
            print(f"\n  Text {i+1} ({len(text)} chars): {text[:30]}...")
            result = benchmark_inference(model, tokenizer, [text], device, num_runs=100)
            gpu_results[f'text_{i+1}_{len(text)}_chars'] = result
            print(f"    Latency: {result['mean_latency_ms']:.2f} ± {result['std_latency_ms']:.2f} ms")
            print(f"    Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")
        
        results['gpu'] = {
            'device': torch.cuda.get_device_name(0),
            'results': gpu_results
        }
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # ========== CPU Benchmark ==========
    print("\n" + "=" * 40)
    print("CPU Benchmark")
    print("=" * 40)
    
    device = torch.device('cpu')
    print(f"Device: CPU ({torch.get_num_threads()} threads)")
    
    # Load model
    model = KoElectraNER(MODEL_NAME, NUM_LABELS)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    cpu_results = {}
    for i, text in enumerate(test_texts):
        print(f"\n  Text {i+1} ({len(text)} chars): {text[:30]}...")
        result = benchmark_inference(model, tokenizer, [text], device, num_runs=50)  # Fewer runs for CPU
        cpu_results[f'text_{i+1}_{len(text)}_chars'] = result
        print(f"    Latency: {result['mean_latency_ms']:.2f} ± {result['std_latency_ms']:.2f} ms")
        print(f"    Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")
    
    results['cpu'] = {
        'device': f'CPU ({torch.get_num_threads()} threads)',
        'results': cpu_results
    }
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print("\n### Latency Comparison (Mean ± Std, ms)")
    print("-" * 50)
    print(f"{'Text Length':<15} | {'GPU':<20} | {'CPU':<20}")
    print("-" * 50)
    
    for key in cpu_results.keys():
        chars = key.split('_')[2]
        cpu_lat = f"{cpu_results[key]['mean_latency_ms']:.2f} ± {cpu_results[key]['std_latency_ms']:.2f}"
        if 'gpu' in results:
            gpu_lat = f"{results['gpu']['results'][key]['mean_latency_ms']:.2f} ± {results['gpu']['results'][key]['std_latency_ms']:.2f}"
        else:
            gpu_lat = "N/A"
        print(f"{chars + ' chars':<15} | {gpu_lat:<20} | {cpu_lat:<20}")
    
    print("\n### Throughput Comparison (samples/sec)")
    print("-" * 50)
    for key in cpu_results.keys():
        chars = key.split('_')[2]
        cpu_tp = f"{cpu_results[key]['throughput_samples_per_sec']:.1f}"
        if 'gpu' in results:
            gpu_tp = f"{results['gpu']['results'][key]['throughput_samples_per_sec']:.1f}"
            speedup = results['gpu']['results'][key]['throughput_samples_per_sec'] / cpu_results[key]['throughput_samples_per_sec']
            print(f"{chars + ' chars':<15}: GPU={gpu_tp:<10} CPU={cpu_tp:<10} Speedup={speedup:.1f}x")
        else:
            print(f"{chars + ' chars':<15}: CPU={cpu_tp}")
    
    # Save results
    with open(f"{OUTPUT_DIR}/benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {OUTPUT_DIR}/benchmark_results.json")


if __name__ == "__main__":
    main()
