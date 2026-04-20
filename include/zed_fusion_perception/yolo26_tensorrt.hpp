#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "zed_fusion_perception/detected_cone.hpp"

class Yolo26nSeg {
public:
    Yolo26nSeg(const std::string& engine_path, float conf_threshold = 0.5, float nms_threshold = 0.45);
    ~Yolo26nSeg();

    // Runs inference and returns parsed detections with masks
    std::vector<DetectedCone> infer(const cv::Mat& bgr_image);

private:
    void loadEngine(const std::string& path);
    void allocateBuffers();
    
    // Prepares the image (Resize, BGR2RGB, HWC2CHW, Normalize)
    void preprocess(const cv::Mat& src, float* dst);
    
    // Handles NMS and Mask Generation
    std::vector<DetectedCone> postprocess(float* output0, float* output1, const cv::Size& original_size);

    // Cuda resources
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    void* buffers_[3]; // Input, Output0, Output1
    
    float conf_threshold_;
    float nms_threshold_;

    // Tensor shapes
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output0_dims_;
    nvinfer1::Dims output1_dims_;

    // Helper for sigmoid
    inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};
