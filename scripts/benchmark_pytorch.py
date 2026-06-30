import os
import sys
import time
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package is required to run PyTorch benchmarks.")
    print("Please install it via: pip install ultralytics")
    sys.exit(1)

def run_benchmark(model_path, source, output_csv="pytorch_stats.csv"):
    print(f"Loading PyTorch model: {model_path}...")
    model = YOLO(model_path)
    
    # Check if source is image folder or video
    is_video = False
    if os.path.isdir(source):
        image_files = [os.path.join(source, f) for f in os.listdir(source) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        if not image_files:
            print(f"No images found in {source}")
            return
        total_frames = len(image_files)
    elif os.path.isfile(source) and source.lower().endswith(('.mp4', '.avi', '.mkv')):
        is_video = True
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        print(f"Invalid source: {source}. Must be an image directory or video file.")
        return

    print(f"Running benchmark on {total_frames} frames...")
    
    latencies = []
    
    # Warmup
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        model(dummy_img, verbose=False)
        
    if is_video:
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.perf_counter()
            results = model(frame, verbose=False)
            end = time.perf_counter()
            
            total_ms = (end - start) * 1000.0
            
            speed = results[0].speed
            preprocess_ms = speed.get('preprocess', 0.0)
            inference_ms = speed.get('inference', 0.0)
            postprocess_ms = speed.get('postprocess', 0.0)
            
            latencies.append({
                'latency_ms': total_ms,
                'preprocess_ms': preprocess_ms,
                'inference_ms': inference_ms,
                'postprocess_ms': postprocess_ms,
                'hz': 1000.0 / total_ms if total_ms > 0 else 0.0
            })
        cap.release()
    else:
        for img_path in tqdm(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            start = time.perf_counter()
            results = model(frame, verbose=False)
            end = time.perf_counter()
            
            total_ms = (end - start) * 1000.0
            
            speed = results[0].speed
            preprocess_ms = speed.get('preprocess', 0.0)
            inference_ms = speed.get('inference', 0.0)
            postprocess_ms = speed.get('postprocess', 0.0)
            
            latencies.append({
                'latency_ms': total_ms,
                'preprocess_ms': preprocess_ms,
                'inference_ms': inference_ms,
                'postprocess_ms': postprocess_ms,
                'hz': 1000.0 / total_ms if total_ms > 0 else 0.0
            })

    df = pd.DataFrame(latencies)
    df.to_csv(output_csv, index=False)
    print(f"\nPyTorch benchmark results saved to {output_csv}")
    print(f"Average Total Latency: {df['latency_ms'].mean():.2f} ms ({1000.0/df['latency_ms'].mean():.1f} FPS)")
    print(f"Split: Preprocess {df['preprocess_ms'].mean():.2f}ms | Inference {df['inference_ms'].mean():.2f}ms | Postprocess {df['postprocess_ms'].mean():.2f}ms")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 benchmark_pytorch.py <model_path.pt> <source_dir_or_video> [output_csv]")
        sys.exit(1)
    
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "pytorch_stats.csv"
    run_benchmark(sys.argv[1], sys.argv[2], out_csv)
