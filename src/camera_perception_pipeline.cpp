#include "camera_perception_pipeline.hpp"
#include <iostream>

namespace camera_perception {

CameraPerceptionPipeline::CameraPerceptionPipeline() {}

CameraPerceptionPipeline::~CameraPerceptionPipeline() {
    if (d_mask_canvas_) {
        cudaFree(d_mask_canvas_);
        d_mask_canvas_ = nullptr;
    }
}

void CameraPerceptionPipeline::initialize(const CameraPipelineConfig& config) {
    yolo_ = std::make_unique<Yolo26nSeg>(
        config.engine_path, 
        config.conf_threshold, 
        config.nms_threshold, 
        config.use_cuda_kernels);
}

std::vector<DetectedCone> CameraPerceptionPipeline::run(
    const cv::Mat& frame, 
    const CameraPipelineConfig& config,
    std::vector<uint8_t>& mask_data_out) {

    std::vector<DetectedCone> detections;
    if (frame.empty() || !yolo_) return detections;

    // 1. Manage device memory buffer dynamically
    size_t required_size = frame.total();
    if (d_mask_canvas_ == nullptr || canvas_size_ != required_size) {
        if (d_mask_canvas_) {
            cudaFree(d_mask_canvas_);
        }
        canvas_size_ = required_size;
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_mask_canvas_), canvas_size_);
        if (err != cudaSuccess) {
            std::cerr << "CUDA error allocating mask canvas: " << cudaGetErrorString(err) << std::endl;
            return detections;
        }
    }

    // 2. Perform inference on GPU
    yolo_->infer_to_canvas(frame, d_mask_canvas_);
    detections = yolo_->infer(frame);

    // 3. Queue download of mask and synchronize CUDA stream
    mask_data_out.resize(canvas_size_);
    cudaMemcpyAsync(
        mask_data_out.data(), 
        d_mask_canvas_, 
        canvas_size_, 
        cudaMemcpyDeviceToHost, 
        yolo_->getStream());

    cudaStreamSynchronize(yolo_->getStream());

    return detections;
}

} // namespace camera_perception
