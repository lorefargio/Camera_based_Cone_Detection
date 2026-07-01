import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_compare(pt_csv, trt_nocuda_csv, trt_cuda_csv):
    files = {
        'PyTorch (.pt)': pt_csv,
        'TensorRT (no CUDA kernels)': trt_nocuda_csv,
        'TensorRT (with CUDA kernels)': trt_cuda_csv
    }

    results = []

    for name, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping {name} in comparison.")
            continue
            
        df = pd.read_csv(filepath)
        df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
        df = df.dropna(subset=['latency_ms'])
        
        if df.empty:
            print(f"Warning: {filepath} contains no valid latency data.")
            continue
            
        avg_latency = df['latency_ms'].mean()
        p95_latency = np.percentile(df['latency_ms'], 95)
        p99_latency = np.percentile(df['latency_ms'], 99)
        std_latency = df['latency_ms'].std()
        avg_fps = df['hz'].mean() if 'hz' in df.columns else (1000.0 / avg_latency if avg_latency > 0 else 0)
        
        # Calculate split timings if available
        avg_pre = df['preprocess_ms'].mean() if 'preprocess_ms' in df.columns else np.nan
        avg_infer = df['inference_ms'].mean() if 'inference_ms' in df.columns else np.nan
        avg_post = df['postprocess_ms'].mean() if 'postprocess_ms' in df.columns else np.nan
        
        results.append({
            'Configuration': name,
            'Avg Latency (ms)': f"{avg_latency:.2f}",
            'P95 (ms)': f"{p95_latency:.2f}",
            'P99 (ms)': f"{p99_latency:.2f}",
            'Preprocess (ms)': f"{avg_pre:.2f}" if not np.isnan(avg_pre) else "N/A",
            'Inference (ms)': f"{avg_infer:.2f}" if not np.isnan(avg_infer) else "N/A",
            'Postprocess (ms)': f"{avg_post:.2f}" if not np.isnan(avg_post) else "N/A",
            'Jitter/Std Dev (ms)': f"{std_latency:.2f}",
            'Avg FPS (Hz)': f"{avg_fps:.1f}"
        })

    if not results:
        print("Error: No stats files found for comparison.")
        return

    df_res = pd.DataFrame(results)
    print("\n" + "="*70)
    print("                      BENCHMARK COMPARISON RESULTS")
    print("="*70)
    print(df_res.to_string(index=False))
    print("="*70)

    # Save to markdown file
    df_res.to_markdown("benchmark_report.md", index=False)
    print("Markdown report saved to benchmark_report.md")

    # Plot comparisons
    plt.figure(figsize=(12, 6))
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            plt.plot(df['latency_ms'], label=name, alpha=0.8, linewidth=2)

    plt.title('Latency Comparison: PyTorch vs TensorRT vs TensorRT+CUDA', fontsize=14, fontweight='bold')
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison chart saved to benchmark_comparison.png")

if __name__ == "__main__":
    pt = 'pytorch_stats.csv' if len(sys.argv) < 2 else sys.argv[1]
    trt_nocuda = 'camera_stats_trt_nocuda.csv' if len(sys.argv) < 3 else sys.argv[2]
    trt_cuda = 'camera_stats_trt_cuda.csv' if len(sys.argv) < 4 else sys.argv[3]
    analyze_and_compare(pt, trt_nocuda, trt_cuda)
