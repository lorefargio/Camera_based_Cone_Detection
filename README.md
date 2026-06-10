# Camera-Based Cone Detection: TensorRT YOLOv8 Segmentation

A high-performance visual perception node designed to detect and segment traffic cones for Formula Student Autonomous racing. The system acts as a high-speed mask provider, supplying precise real-time semantic segmentation maps and bounding boxes for downstream LiDAR-Camera fusion.

---

## 1. Decoupled Architecture

Following our strict engineering standards, the node is structured using decoupled C++ classes to separate ROS 2 middleware from GPU calculation and visualization serialization:

```
[Incoming Image msg] --> (ZedPerceptionNode ROS Wrapper)
                                   |
                             (cv::Mat Frame)
                                   v
                    (CameraPerceptionPipeline Facade)
                                   |
                  (TensorRT Inference & CUDA Kernels)
                                   |
                         (Detections & Canvas)
                                   v
                (CameraVisualizationBridge Active Object)
                                   |
                     (Asynchronous drawing & pub)
```

---

## 2. Technical Documentation Index

For detailed theoretical derivations, CUDA performance details, and concurrency models, refer to:
*   **[Pipeline Architecture](docs/architecture.md)**: Details the stage-by-stage data flow, facade layout, and active visualization bridge.
*   **[CUDA Pipeline Optimization](docs/cuda_optimizations.md)**: Deep dive into the custom GPU preprocessing kernel, HWC memory transposition (solving L1/L2 cache misses), and Shared Memory Tiling for mask generation.
*   **[Concurrency & Stream Synchronization](docs/concurrency.md)**: Explains the multi-threaded execution layout (Active Object) and the single-barrier CUDA stream synchronization (`cudaStreamSynchronize`) path.

---

## 3. Build Dependencies

Ensure your ROS 2 environment contains the following packages and system libraries before compilation:
*   **ROS 2 Distribution**: Humble (or newer)
*   **OpenCV**: Image processing library (`libopencv-dev`)
*   **CUDA Toolkit**: Required for preprocessing, reformatting, and postprocessing GPU kernels.
*   **TensorRT**: High-performance deep learning inference library.
*   **ROS Packages**: `rclcpp`, `rclcpp_components`, `sensor_msgs`, `vision_msgs`, `geometry_msgs`, `visualization_msgs`, `cv_bridge`, `image_transport`, `tf2_ros`, `tf2_geometry_msgs`.

---

## 4. Compile & Run Instructions

### Step 1: Compile the Package
Compile the package inside your workspace with Release optimizations:
```bash
colcon build --packages-select camera_perception --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install
```

### Step 2: Source Setup Script
```bash
source install/setup.zsh
```

### Step 3: Launch Node
Launch the camera perception component node:
```bash
ros2 launch camera_perception test_detection_launch.py \
    use_cuda_kernels:=true \
    publish_debug:=true
```

### Step 4: Run Latency Benchmarks
Compare custom CUDA kernels against CPU-only baseline:
```bash
# 1. Run with CUDA Kernels enabled
ros2 launch camera_perception test_detection_launch.py export_stats:=true use_cuda_kernels:=true

# 2. Run with CPU baseline (downloads raw prototypes and does CPU postprocessing)
ros2 launch camera_perception test_detection_launch.py export_stats:=true use_cuda_kernels:=false

# 3. Plot performance comparison
python3 scripts/analyze_performance.py
```
