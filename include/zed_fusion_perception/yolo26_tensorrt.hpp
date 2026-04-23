#pragma once
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include "zed_fusion_perception/detected_cone.hpp"

extern "C" void launch_preprocess(const uint8_t* src, void* dst, int src_w, int src_h, int dst_w, int dst_h, int channels, bool is_fp16, cudaStream_t stream);
extern "C" void launch_reformat_prototypes(const void* src, void* dst, int w, int h, int channels, bool is_fp16, cudaStream_t stream);
extern "C" void launch_postprocess_mask(const void* output0, const void* output1, uint8_t* mask_canvas, int canvas_w, int canvas_h, float conf_threshold, bool is_fp16, cudaStream_t stream);

class Yolo26nSeg {
public:
    Yolo26nSeg(const std::string& engine_path, float conf_threshold = 0.5, float nms_threshold = 0.45);
    ~Yolo26nSeg();

    // Standard inference (returns detections on CPU)
    std::vector<DetectedCone> infer(const cv::Mat& bgr_image);
    
    // High-performance: populates the d_mask_canvas on GPU
    void infer_to_canvas(const cv::Mat& bgr_image, uint8_t* d_mask_canvas);

    cudaStream_t getStream() const { return stream_; }

private:
    void loadEngine(const std::string& path);
    void allocateBuffers();
    void preprocess_gpu(const cv::Mat& src);
    std::vector<DetectedCone> postprocess(void* output0, void* output1, const cv::Size& original_size);

    // CUDA Graph management
    void initGraph(const cv::Mat& sample_img, uint8_t* d_mask_canvas);

    // Cuda resources
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;
    void* buffers_[3]; 
    void* d_src_image_; 
    void* d_proto_reformatted_; 
    
    // CUDA Graph members
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
    bool graph_initialized_;

    float conf_threshold_;
    float nms_threshold_;
    bool is_fp16_;

    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output0_dims_;
    nvinfer1::Dims output1_dims_;

    std::string input_name_;
    std::string output0_name_;
    std::string output1_name_;

    std::vector<char> host_output0_raw_;
    std::vector<char> host_output1_raw_;

    inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};
