# CUDA Pipeline Optimization and Custom Kernels

This document explains the custom CUDA kernels and memory optimizations implemented in the camera perception stack to achieve real-time inference rates exceeding 100Hz on edge computing hardware.

---

## 1. Custom GPU Preprocessing Kernel

Traditional CPU-side preprocessing (resizing, normalizing, and converting a BGR image layout to planar FP32 format) is extremely slow and acts as a serialization bottleneck. We offload this to the GPU using a dedicated CUDA kernel.

### `launch_preprocess`
This kernel takes a raw BGR image in host memory and processes it on the GPU:
*   **Bilinear Resizing**: Interlaced resizing to fit the YOLO model's input dimensions (e.g. $640 \times 640$).
*   **Color Conversion & Layout Normalization**: Converts BGR to RGB, scales pixel values from $[0, 255]$ to $[0.0, 1.0]$, and reformats the memory from **HWC** to **CHW** layout required by the model.
*   **Planar Conversion**: Employs FP16 or FP32 depending on engine specifications.

---

## 2. Resolving the CHW Cache locality Bottleneck (Transposition)

Deep Learning models generate mask prototype tensors in **CHW** (Channel-Height-Width) format. In this layout, the channel values for a single spatial pixel $(x, y)$ are distributed at physical memory distances equal to $H \times W$.

```
CHW Memory Layout (Sparse):
[Channel 0: Pixel 0, 1, 2...] ... [Channel 1: Pixel 0, 1, 2...] ... [Channel 31: Pixel 0, 1...]
```

During post-processing, computing the binary mask requires calculating a dot product between 32 coefficients and the 32 prototype channels for *every* pixel:
$$\text{Mask}(x, y) = \sigma \left( \sum_{c=0}^{31} \text{coeff}_c \times \text{proto}(x, y, c) \right)$$

In CHW layout, loading these 32 channels requires requesting 32 distinct memory addresses separated by thousands of bytes. This causes **non-coalesced memory transactions** and **saturates the PCIe/VRAM bandwidth**.

### `launch_reformat_prototypes` (CHW to HWC)
We introduce an optimized transposition kernel that reorganizes the prototype tensor into **HWC** format on the GPU.

```
HWC Memory Layout (Contiguous):
[Pixel 0: Channel 0, 1... 31] [Pixel 1: Channel 0, 1... 31] ...
```

In HWC format, the 32 channels for pixel $(x, y)$ are stored **physically adjacent** in memory. When a thread accesses the first channel, the GPU's memory controller loads the entire 32-channel block into the L1/L2 cache in a single coalesced transaction, drastically reducing memory latency.

---

## 3. Shared Memory Tiling for Mask Generation

### `launch_postprocess_mask`
Without optimization, each GPU thread (one per canvas pixel) must read the 32 coefficients for every bounding box detection from Global Memory, resulting in $\mathcal{O}(\text{Pixels} \times \text{Detections})$ global memory reads.

We use **Shared Memory Tiling** to eliminate this redundancy:
1.  **Cooperative Loading**: At the start of a thread block, threads cooperatively load the bounding boxes and 32 coefficients of the top 128 detections from Global Memory into the block's fast on-chip **Shared Memory**.
2.  **Synchronization Barrier**: The block calls `__syncthreads()` to ensure all coefficients are loaded.
3.  **Low-Latency Access**: All threads in the block access the coefficients from Shared Memory with near-zero latency, reducing Global Memory access to $\mathcal{O}(\text{Blocks} \times \text{Detections})$.

---

## 4. Pipeline Memory Lifecycle Management

Historically, CUDA memory buffers for mask canvasing were allocated as static global pointers inside the callback function. 
In the refactored [camera_perception_pipeline.cpp](file:///Users/m2pro/Lidar_Camera_fusion_ws/src/Camera_based_Cone_Detection/src/camera_perception_pipeline.cpp), this is refactored to conform to class resource allocation principles (RAII):

*   **RAII Allocation**: Memory buffers (`d_mask_canvas_`) are owned by the `CameraPerceptionPipeline` instance.
*   **Dynamic Reallocation**: If the incoming image resolution changes (e.g. changing camera mode), the pipeline dynamically frees the old buffer via `cudaFree` and allocates a new one via `cudaMalloc` in-place.
*   **Destructor Cleanup**: The destructor safely cleans up all allocated GPU pointers, preventing memory leaks on node shutdown.
