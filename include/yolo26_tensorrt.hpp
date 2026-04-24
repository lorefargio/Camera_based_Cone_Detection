#pragma once
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include "detected_cone.hpp"

// Forward declarations for CUDA kernels
extern "C" void launch_preprocess(const uint8_t* src, void* dst, int src_w, int src_h, int dst_w, int dst_h, int channels, bool is_fp16, cudaStream_t stream);
extern "C" void launch_reformat_prototypes(const void* src, void* dst, int w, int h, int channels, bool is_fp16, cudaStream_t stream);
extern "C" void launch_postprocess_mask(const void* output0, const void* output1, uint8_t* mask_canvas, int canvas_w, int canvas_h, float conf_threshold, bool is_fp16, cudaStream_t stream);

/**
 * @brief High-performance YOLO26n-Seg inference engine using TensorRT.
 * 
 * This class manages the TensorRT engine lifecycle and executes the optimized
 * GPU pipeline, including preprocessing, inference, and semantic mask generation.
 */
class Yolo26nSeg {
public:
    /**
     * @brief Constructor for Yolo26nSeg.
     * @param engine_path Path to the serialized TensorRT engine file.
     * @param conf_threshold Confidence threshold for filtering detections.
     * @param nms_threshold Non-Maximum Suppression threshold.
     */
    Yolo26nSeg(const std::string& engine_path, float conf_threshold = 0.5, float nms_threshold = 0.45);
    
    /**
     * @brief Destructor for Yolo26nSeg. Cleans up CUDA and TensorRT resources.
     */
    ~Yolo26nSeg();

    /**
     * @brief Performs standard inference and returns a list of detections on CPU.
     * @param bgr_image Input BGR image from the camera.
     * @return std::vector<DetectedCone> List of detected cones.
     */
    std::vector<DetectedCone> infer(const cv::Mat& bgr_image);
    
    /**
     * @brief High-performance path: populates a semantic mask canvas directly on the GPU.
     * @param bgr_image Input BGR image from the camera.
     * @param d_mask_canvas Device pointer to the output mono8 mask canvas.
     */
    void infer_to_canvas(const cv::Mat& bgr_image, uint8_t* d_mask_canvas);

    /**
     * @brief Gets the CUDA stream used by the engine for external synchronization.
     * @return cudaStream_t The internal CUDA stream.
     */
    cudaStream_t getStream() const { return stream_; }

private:
    /**
     * @brief Deserializes and loads the TensorRT engine.
     * @param path Path to the engine file.
     */
    void loadEngine(const std::string& path);

    /**
     * @brief Allocates GPU memory for inputs, outputs, and intermediate buffers.
     */
    void allocateBuffers();

    /**
     * @brief Prepares the input image on the GPU (resizing and normalization).
     * @param src Source BGR image.
     */
    void preprocess_gpu(const cv::Mat& src);

    /**
     * @brief CPU side post-processing to parse raw TensorRT output into DetectedCone objects.
     * @param output0 Pointer to the primary detection output.
     * @param output1 Pointer to the mask prototypes output (optional).
     * @param original_size Size of the original input image for coordinate scaling.
     * @return std::vector<DetectedCone> Parsed detections.
     */
    std::vector<DetectedCone> postprocess(void* output0, void* output1, const cv::Size& original_size);

    // TensorRT resources
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // CUDA resources
    cudaStream_t stream_;
    void* buffers_[3];            ///< GPU buffers for TensorRT (Input, Output0, Output1)
    void* d_src_image_;           ///< Device pointer for the raw source image
    void* d_proto_reformatted_;   ///< Device pointer for the HWC-reformatted prototypes
    
    float conf_threshold_;
    float nms_threshold_;
    bool is_fp16_;

    // Tensor shapes and names
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output0_dims_;
    nvinfer1::Dims output1_dims_;
    std::string input_name_;
    std::string output0_name_;
    std::string output1_name_;

    // Host buffers for asynchronous download
    std::vector<char> host_output0_raw_;
    std::vector<char> host_output1_raw_;

    /**
     * @brief Standard sigmoid activation function.
     */
    inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};
