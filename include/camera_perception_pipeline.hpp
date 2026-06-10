#pragma once

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include "yolo26_tensorrt.hpp"
#include "camera_pipeline_config.hpp"

namespace camera_perception {

/**
 * @class CameraPerceptionPipeline
 * @brief Facade class executing GPU pre-processing, TensorRT YOLO inference, and mask copying.
 */
class CameraPerceptionPipeline {
public:
    CameraPerceptionPipeline();
    ~CameraPerceptionPipeline();

    /**
     * @brief Instantiates the TensorRT YOLO engine.
     */
    void initialize(const CameraPipelineConfig& config);
    
    /**
     * @brief Executes TensorRT inference, populates the GPU canvas mask, and copies the mask back to host.
     */
    std::vector<DetectedCone> run(
        const cv::Mat& frame, 
        const CameraPipelineConfig& config,
        std::vector<uint8_t>& mask_data_out);

    /**
     * @brief Returns the underlying CUDA stream for synchronization.
     */
    cudaStream_t getStream() const { return yolo_ ? yolo_->getStream() : nullptr; }

private:
    std::unique_ptr<Yolo26nSeg> yolo_;
    uint8_t* d_mask_canvas_ = nullptr;
    size_t canvas_size_ = 0;
};

} // namespace camera_perception
