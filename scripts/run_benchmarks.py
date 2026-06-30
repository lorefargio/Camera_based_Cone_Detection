import os
import sys
import time
import subprocess
import shutil

def run_ros2_node_test(bag_path, is_pytorch, engine_path, use_cuda, dest_csv):
    mode_name = "PyTorch (.pt)" if is_pytorch else f"TRT Engine (use_cuda={use_cuda})"
    print(f"\n" + "="*70)
    print(f">>> STARTING BENCHMARK: {mode_name}")
    print("="*70)
    
    if is_pytorch:
        # Start the ROS 2 Python PyTorch node
        launch_cmd = [
            "python3", "scripts/pytorch_perception_node.py",
            engine_path,  # Passed pt path as engine_path parameter here
            dest_csv
        ]
    else:
        # Start the ROS 2 C++ TensorRT node
        launch_cmd = [
            "ros2", "launch", "camera_perception", "test_detection_launch.py",
            f"engine_path:={engine_path}",
            f"use_cuda_kernels:={str(use_cuda).lower()}",
            "export_stats:=true",
            "publish_debug:=false"
        ]
    
    print(f"Launching node: {' '.join(launch_cmd)}")
    node_proc = subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for the node to initialize/warm up
    time.sleep(6.0 if is_pytorch else 5.0)
    
    # Start the ROS bag playback (blocks until complete)
    bag_cmd = ["ros2", "bag", "play", bag_path, "--rate", "1.0"]
    print(f"Playing ROS bag: {' '.join(bag_cmd)}")
    try:
        subprocess.run(bag_cmd, check=True)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    
    # Clean up processes
    print("Stopping nodes...")
    node_proc.terminate()
    node_proc.wait()
    
    if not is_pytorch:
        # For C++ node, move the output CSV to its destination
        src_csv = "camera_stats.csv"
        if os.path.exists(src_csv):
            shutil.move(src_csv, dest_csv)
            print(f"Saved C++ stats to: {dest_csv}")
        else:
            print(f"Error: {src_csv} was not generated. Did the node fail?")
    else:
        print(f"Saved PyTorch stats to: {dest_csv}")

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
        
    # Run Test 1: TRT without CUDA
    run_ros2_node_test(bag_path, is_pytorch=False, engine_path=engine_path, use_cuda=False, dest_csv="camera_stats_trt_nocuda.csv")
    
    # Run Test 2: TRT with CUDA
    run_ros2_node_test(bag_path, is_pytorch=False, engine_path=engine_path, use_cuda=True, dest_csv="camera_stats_trt_cuda.csv")
    
    # Run Test 3: PyTorch Model via ROS 2 Python Node
    if os.path.exists(pt_path):
        run_ros2_node_test(bag_path, is_pytorch=True, engine_path=pt_path, use_cuda=False, dest_csv="pytorch_stats.csv")
    else:
        print(f"PyTorch model not found at {pt_path}. Skipping PyTorch benchmark.")
        
    # Run comparison and plotting
    print("\n" + "="*70)
    print(">>> COMPILING RESULTS")
    print("="*70)
    compare_cmd = ["python3", "scripts/compare_performance.py", "pytorch_stats.csv", "camera_stats_trt_nocuda.csv", "camera_stats_trt_cuda.csv"]
    subprocess.run(compare_cmd)

if __name__ == "__main__":
    main()
