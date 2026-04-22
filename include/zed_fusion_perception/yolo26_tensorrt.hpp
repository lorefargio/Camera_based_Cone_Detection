#pragma once
#include <vector>
#include <string>
#include <memory>
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
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;
    void* buffers_[3]; // Input, Output0, Output1
    
    float conf_threshold_;
    float nms_threshold_;

    // Tensor shapes
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output0_dims_;
    nvinfer1::Dims output1_dims_;

    // Tensor names (Required for TensorRT 10 enqueueV3)
    std::string input_name_;
    std::string output0_name_;
    std::string output1_name_;

    std::vector<float> host_input_batch_;
    std::vector<float> host_output0_;
    std::vector<float> host_output1_;

    // Helper for sigmoid
    inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};
