import os
import sys
import time
import subprocess
import shutil

def run_test(bag_path, engine_path, use_cuda, dest_csv, extract_frames=False):
    print(f"\n" + "="*60)
    print(f">>> STARTING BENCHMARK: TRT Engine (use_cuda_kernels={use_cuda})")
    print("="*60)
    
    # 1. Start the ROS 2 perception node
    launch_cmd = [
        "ros2", "launch", "camera_perception", "test_detection_launch.py",
        f"engine_path:={engine_path}",
        f"use_cuda_kernels:={str(use_cuda).lower()}",
        "export_stats:=true",
        "publish_debug:=false"
    ]
    
    print(f"Launching node: {' '.join(launch_cmd)}")
    node_proc = subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for the node and TensorRT engine to initialize
    time.sleep(5.0)
    
    # 2. If frame extraction is requested, launch the extractor in the background
    extractor_proc = None
    if extract_frames:
        print("Starting frame extractor node in background...")
        extractor_cmd = ["python3", "scripts/extract_frames.py"]
        extractor_proc = subprocess.Popen(extractor_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
        
    # 3. Start the ROS bag playback (blocks until complete)
    bag_cmd = ["ros2", "bag", "play", bag_path, "--rate", "1.0"]
    print(f"Playing ROS bag: {' '.join(bag_cmd)}")
    try:
        subprocess.run(bag_cmd, check=True)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    
    # 4. Clean up processes
    print("Stopping nodes...")
    if extractor_proc:
        extractor_proc.terminate()
        extractor_proc.wait()
        
    node_proc.terminate()
    node_proc.wait()
    
    # 5. Move output CSV to destination
    src_csv = "camera_stats.csv"
    if os.path.exists(src_csv):
        shutil.move(src_csv, dest_csv)
        print(f"Saved stats to: {dest_csv}")
    else:
        print(f"Error: {src_csv} was not generated. Did the node crash or fail to run?")

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 run_benchmarks.py <bag_path> <engine_path> <pt_model_path>")
        sys.exit(1)
        
    bag_path = os.path.abspath(sys.argv[1])
    engine_path = os.path.abspath(sys.argv[2])
    pt_path = os.path.abspath(sys.argv[3])
    
    # Check running directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(script_dir)
    os.chdir(workspace_dir)
    print(f"Running benchmarks inside workspace: {workspace_dir}")
        
    # Create output directory for temporary extracted frames
    frames_dir = "extracted_frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        
    # Run Test 1: TRT without CUDA (and extract frames for PyTorch)
    run_test(bag_path, engine_path, use_cuda=False, dest_csv="camera_stats_trt_nocuda.csv", extract_frames=True)
    
    # Run Test 2: TRT with CUDA
    run_test(bag_path, engine_path, use_cuda=True, dest_csv="camera_stats_trt_cuda.csv")
    
    # Run Test 3: PyTorch Model benchmark
    print("\n" + "="*60)
    print(">>> STARTING BENCHMARK: PyTorch Model (.pt)")
    print("="*60)
    if os.path.exists(pt_path):
        pt_cmd = ["python3", "scripts/benchmark_pytorch.py", pt_path, frames_dir, "pytorch_stats.csv"]
        print(f"Running command: {' '.join(pt_cmd)}")
        subprocess.run(pt_cmd)
    else:
        print(f"PyTorch model not found at {pt_path}. Skipping PyTorch benchmark.")
        
    # Run comparison and plotting
    print("\n" + "="*60)
    print(">>> COMPILING RESULTS")
    print("="*60)
    compare_cmd = ["python3", "scripts/compare_performance.py", "pytorch_stats.csv", "camera_stats_trt_nocuda.csv", "camera_stats_trt_cuda.csv"]
    subprocess.run(compare_cmd)

if __name__ == "__main__":
    main()
