# Concurrency, Threading, and Stream Synchronization

To sustain processing rates exceeding 100Hz and prevent pipeline lag, the camera node utilizes a multi-threaded execution model combined with a single-barrier CUDA stream synchronization design.

---

## 1. Multi-Threaded Task Offloading (Active Object)

The camera perception node splits execution across two dedicated threads:

```
[Camera Image Sub] ---> (Main Executor Thread) ---> [Inference Facade]
                                                            |
                                                   (Enqueue Snapshot)
                                                            v
[Publishers] <--------- (Visualization Thread) <--- [Thread-Safe Queue]
```

### 1.1 The Main Executor Thread
This thread is managed by the ROS 2 executor. It handles message ingress and execution:
1.  Converts the ROS Image message to an OpenCV `cv::Mat` frame.
2.  Invokes `CameraPerceptionPipeline::run()`. This schedules all GPU tasks (preprocessing, inference, postprocessing) and triggers an asynchronous device-to-host copy of the mask data.
3.  Crosses a single CUDA stream barrier (`cudaStreamSynchronize`) to ensure the host mask memory vector is fully populated.
4.  Enqueues the frame snapshot (BGR image, host mask vector, and 2D bounding box detections) into the `CameraVisualizationBridge` task queue, then returns immediately.

### 1.2 The Asynchronous Visualization Thread (`VisualizationBridge`)
Spins a dedicated background worker thread that processes visual outputs:
*   **Decoupled Rendering**: Builds the debug images (drawing center circles and box edges via OpenCV `cv::circle`) and formats the RViz bounding box markers.
*   **Decoupled Serialization**: Converts raw host memory buffers to ROS `sensor_msgs::msg::Image` messages and publishes them to the middleware.
*   **Benefit**: Drawing and publishing takes 3-5ms. Offloading this ensures the main thread is freed immediately to receive the next high-frequency camera frame.

---

## 2. Single-Barrier Stream Synchronization

All GPU calculations are scheduled asynchronously on a single CUDA execution stream (`yolo_->getStream()`). This prevents host-device CPU-GPU stalls during execution.

```
CUDA Stream Timeline:
[ launch_preprocess ] ---> [ TensorRT Execute ] ---> [ launch_postprocess_mask ] ---> [ cudaMemcpyAsync ] -| (Sync Barrier)
```

1.  **Pipeline Launch**: The kernels (`launch_preprocess`, TensorRT inference engine execution, and `launch_postprocess_mask`) are submitted sequentially to the stream. The GPU queues and runs them back-to-back.
2.  **Asynchronous Copy**: We submit `cudaMemcpyAsync` to copy the final mask canvas from the device buffer (`d_mask_canvas_`) to host memory.
3.  **The Synchronization Barrier**: The host calls `cudaStreamSynchronize()`. This blocks the CPU thread until all queued GPU kernels and the memory download are complete.
4.  **Why a Single Barrier?**
    By queuing all operations (preprocessing, inference, postprocessing, copy) before calling a synchronization block, we allow the GPU to pipeline execution seamlessly, avoiding intermediate CPU-GPU round-trip stalls.
