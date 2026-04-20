# Status Report: ZED-LiDAR Fusion Perception Node

## 1. Current Development Status
The repository contains a complete implementation of the ZED2i + LiDAR fusion pipeline optimized for YOLO26n-seg using TensorRT.

### Implemented Components:
- **`Yolo26nSeg` Class**: Handles TensorRT engine loading, CUDA buffer management, image preprocessing (resize/normalize), and advanced post-processing (NMS + Mask reconstruction from prototypes).
- **`SensorFusionEngine`**: Implements the 100/80/20 confidence logic. Matches camera detections with LiDAR clusters using 2D Euclidean distance.
- **`ZedPerceptionNode`**: ROS 2 node subscribing to ZED RGB/Info and LiDAR clusters. Features real-time performance logging.
- **Testing Tools**: Standalone C++ test for YOLO inference and a structured ROS 2 launch file.

---

## 2. Performance Monitoring & P99 Analysis
The node is instrumented with high-precision timers (`std::chrono::high_resolution_clock`). Every frame processed outputs a log line in the following format:

`[INFO] PERF: YOLO: 12.50ms | Fusion: 0.45ms | Total: 15.20ms | Detected: 8`

### How to analyze P99 Performance:
To extract statistics from a ROS bag run, you can pipe the output to a CSV or use a simple python script to parse the logs:

1. **Run the node and capture logs**:
   ```bash
   ros2 launch zed_fusion_perception perception_launch.py | tee perception_perf.log
   ```

2. **Calculate Percentiles (Bash/Python snippet)**:
   ```python
   # Simple Python snippet to calculate P99 from log
   import numpy as np
   latencies = []
   with open('perception_perf.log', 'r') as f:
       for line in f:
           if "Total:" in line:
               latencies.append(float(line.split("Total: ")[1].split("ms")[0]))
   print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
   ```

---

## 3. Testing and Visualization

### A. Testing with ROS Bags
If you have a recorded bag containing ZED topics (`/zed2i/zed_node/rgb/image_rect_color` and `/zed2i/zed_node/rgb/camera_info`):

1. **Launch the Perception Node**:
   ```bash
   ros2 launch zed_fusion_perception test_detection_launch.py use_foxglove:=true
   ```

2. **Play the Bag**:
   ```bash
   ros2 bag play path/to/your_bag.mcap
   ```

### B. Testing with Live ZED Camera
To test in real-time with the camera connected:

1. **Launch everything**:
   ```bash
   ros2 launch zed_fusion_perception test_detection_launch.py use_zed:=true use_foxglove:=true
   ```
   *(Note: This requires the `zed_wrapper` package to be installed.)*

### C. Visualization with Foxglove (Recommended)
Foxglove provides the best experience for visualizing masks and 3D detections.

1. Open [Foxglove Studio](https://studio.foxglove.dev).
2. Connect to `ws://localhost:8765` (requires `foxglove_bridge` which is started by the launch file).
3. **Import Layout**: Open the `foxglove_layout.json` file provided in this repository to automatically set up the Image and 3D views.
4. You should see:
   - **`/perception/debug_image`**: RGB stream with bounding boxes and colorized masks.
   - **`/perception/markers`**: 3D cylinders representing the detected cones in the camera frame.

### D. Visualization with RViz2
1. Run RViz2: `rviz2`.
2. Add an **Image** display for topic `/perception/debug_image`.
3. Add a **MarkerArray** display for topic `/perception/markers`.
4. Set the **Fixed Frame** to `zed2i_left_camera_optical_frame`.

---

## 4. Performance Monitoring & P99 Analysis
... (rest of the content)
- **Coordinate Frames**: Ensure `tf2` transforms between `zed2i_left_camera_frame` and `lidar_frame` are published.
- **Mask Bleed**: Post-processing currently crops masks to bounding boxes to prevent artifacts.
- **Calibration**: The `projectTo3D` function uses a flat-ground assumption as a fallback. For better results, integrate the ZED depth map or rely entirely on LiDAR spatial mapping.
