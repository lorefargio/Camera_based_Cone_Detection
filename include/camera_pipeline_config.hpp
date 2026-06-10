#pragma once

#include <string>

namespace camera_perception {

/**
 * @struct CameraPipelineConfig
 * @brief Structured configuration containing all hyperparameters for the Camera perception pipeline.
 */
struct CameraPipelineConfig {
    std::string engine_path = "models/yolo26n-seg.engine";
    double conf_threshold = 0.5;
    double nms_threshold = 0.45;
    bool publish_debug = false;
    bool export_stats = false;
    bool use_cuda_kernels = true;
};

} // namespace camera_perception
