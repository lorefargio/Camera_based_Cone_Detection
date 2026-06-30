import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def analyze(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
    df = df.dropna(subset=['latency_ms'])
    
    if df.empty:
        print(f"Error: {csv_file} contains no valid latency data.")
        return

    # Calculate stats
    avg_latency = df['latency_ms'].mean()
    p99_latency = np.percentile(df['latency_ms'], 99)
    std_latency = df['latency_ms'].std()
    avg_hz = df['hz'].mean()

    print(f"--- Performance Analysis ---")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"99th Percentile: {p99_latency:.2f} ms")
    print(f"Std Deviation:  {std_latency:.2f} ms")
    print(f"Average FPS:    {avg_hz:.1f} Hz")
    print(f"----------------------------")

    # Create Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Latency over time
    ax1.plot(df['latency_ms'], label='Latency (ms)', color='blue', alpha=0.7)
    ax1.axhline(y=p99_latency, color='red', linestyle='--', label=f'P99 ({p99_latency:.2f}ms)')
    ax1.set_title('Perception Latency over Time')
    ax1.set_ylabel('ms')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Histogram of latency (Jitter analysis)
    ax2.hist(df['latency_ms'], bins=30, color='green', alpha=0.6, edgecolor='black')
    ax2.set_title('Latency Distribution (Jitter)')
    ax2.set_xlabel('ms')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('performance_plots.png')
    print("Plots saved to performance_plots.png")
    plt.show()

if __name__ == "__main__":
    file = 'camera_stats.csv' if len(sys.argv) < 2 else sys.argv[1]
    analyze(file)
