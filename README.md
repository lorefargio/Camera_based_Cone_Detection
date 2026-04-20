# ZED Camera-Based Cone Detection (YOLO26n-Seg)

This repository implements a high-performance perception node using **YOLO26n-seg** with **TensorRT** optimization. It is designed to be the primary visual source for a LiDAR-Camera fusion pipeline in Formula Student Driverless environments.

## 1. Perception Logic & Fusion Strategy

The core philosophy of this node is **"Geometry over Proximity"**. Instead of providing simple bounding boxes, we provide instance-level segmentation masks.

### The "Point-in-Mask" Bridge
Classic fusion uses Bounding Box proximity, which is prone to errors (e.g., a LiDAR point of the ground or a nearby cone falling inside a large BBox). 
Our strategy uses a **Mask Canvas**:
1.  **YOLO Node**: Detects cones and generates a `mono8` image (Mask Canvas) where each pixel's value corresponds to the `detection_id` of the cone.
2.  **Fusion Node**: Projects 3D LiDAR clusters onto the camera's image plane.
3.  **Validation**: A LiDAR cluster is confirmed as a cone ONLY if a significant percentage of its projected points fall on pixels with the correct ID in the Mask Canvas.

### Confidence Model (100/80/20)
- **Confidence 1.0 (LiDAR Only)**: High spatial precision, but no color information or class validation.
- **Confidence 0.8 (Fused)**: The "Gold Standard". LiDAR provides the (x,y,z) and YOLO provides the color/class via the Point-in-Mask match.
- **Confidence 0.2 (Camera Only)**: YOLO sees a cone but LiDAR doesn't. Likely a "ghost" detection or a cone beyond LiDAR range (fallback to pinhole projection).

## 2. Zero-Copy & High Performance
To achieve sub-15ms latency on Jetson Orin/Xavier:
- **Intra-Process Communication (IPC)**: By using `std::unique_ptr` in publishers and `toCvShare` in subscribers, ROS 2 passes pointers instead of copying large image buffers.
- **TensorRT Optimization**: Uses FP16 inference and pre-allocated CUDA memory pools to avoid runtime allocation overhead.
- **Mask Canvas Compression**: The mask is published as a single-channel `mono8` image, minimizing bandwidth.

## 3. Step-by-Step Data Flow
1.  **Input**: `sensor_msgs/Image` from ZED2i.
2.  **Inference**: TensorRT processes the image, outputting class IDs, confidence, and binary masks.
3.  **Encoding**: Each mask is drawn onto a `mask_canvas` using its index (1, 2, 3...) as the pixel value.
4.  **Publishing**:
    - `/perception/debug_image`: For humans (colorized masks + boxes).
    - `/perception/camera_mask_canvas`: For the Fusion Node (The "ID map").
    - `/perception/markers`: 3D cylinders for RViz/Foxglove.

## 4. Setup & Testing
Refer to [STATUS_AND_TESTING.md](STATUS_AND_TESTING.md) for detailed instructions on building, running with rosbags, and Foxglove visualization.
