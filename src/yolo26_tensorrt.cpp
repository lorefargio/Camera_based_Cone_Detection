#include "yolo26_tensorrt.hpp"
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

/**
 * @brief Logger for TensorRT error and warning messages.
 */
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

Yolo26nSeg::Yolo26nSeg(const std::string &engine_path, float conf_threshold,
                       float nms_threshold, bool use_cuda_kernels)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold), use_cuda_kernels_(use_cuda_kernels) {
  loadEngine(engine_path);
  allocateBuffers();
  cudaStreamCreate(&stream_);
}

Yolo26nSeg::~Yolo26nSeg() {
  cudaStreamDestroy(stream_);
  for (void *buf : buffers_) cudaFree(buf);
  cudaFree(d_src_image_);
  cudaFree(d_proto_reformatted_);
}

void Yolo26nSeg::loadEngine(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) throw std::runtime_error("Failed to read engine file: " + path);
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> data(size);
  file.read(data.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
  context_.reset(engine_->createExecutionContext());

  // Determine if the engine uses FP16 precision
  is_fp16_ = false;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string name = engine_->getIOTensorName(i);
    nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
    nvinfer1::DataType type = engine_->getTensorDataType(name.c_str());
    if (type == nvinfer1::DataType::kHALF) is_fp16_ = true;
    
    // Identify input and output tensors by shape
    if (dims.nbDims == 4 && dims.d[1] == 3) {
      input_name_ = name; input_dims_ = dims;
    } else if (dims.nbDims == 3) {
      output0_name_ = name; output0_dims_ = dims;
    } else if (dims.nbDims == 4 && dims.d[1] == 32) {
      output1_name_ = name; output1_dims_ = dims;
    }
  }
}

void Yolo26nSeg::allocateBuffers() {
  auto get_size = [](nvinfer1::Dims dims) {
    size_t s = 1;
    for (int i = 0; i < dims.nbDims; ++i) s *= dims.d[i];
    return s;
  };
  size_t element_size = is_fp16_ ? 2 : 4;
  cudaMalloc(&buffers_[0], get_size(input_dims_) * element_size);
  cudaMalloc(&buffers_[1], get_size(output0_dims_) * element_size);
  cudaMalloc(&buffers_[2], get_size(output1_dims_) * element_size);
  cudaMalloc(&d_src_image_, 2208 * 1242 * 4); // Max ZED 2k resolution
  cudaMalloc(&d_proto_reformatted_, get_size(output1_dims_) * element_size);
  host_output0_raw_.resize(get_size(output0_dims_) * element_size);
  host_output1_raw_.resize(get_size(output1_dims_) * element_size);
}

void Yolo26nSeg::preprocess_cpu(const cv::Mat& bgr_image) {
  cv::Mat resized;
  cv::resize(bgr_image, resized, cv::Size(input_dims_.d[3], input_dims_.d[2]));
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  
  size_t vol = input_dims_.d[1] * input_dims_.d[2] * input_dims_.d[3];
  if (is_fp16_) {
    std::vector<uint16_t> host_input(vol);
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < input_dims_.d[2] * input_dims_.d[3]; ++i) {
        float val = rgb.data[i * 3 + c] / 255.0f;
        // Simple float32 to float16 conversion for comparison purposes
        uint32_t x = *(uint32_t*)&val;
        uint16_t h = ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
        host_input[c * (input_dims_.d[2] * input_dims_.d[3]) + i] = h;
      }
    }
    cudaMemcpyAsync(buffers_[0], host_input.data(), vol * 2, cudaMemcpyHostToDevice, stream_);
  } else {
    std::vector<float> host_input(vol);
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < input_dims_.d[2] * input_dims_.d[3]; ++i) {
        host_input[c * (input_dims_.d[2] * input_dims_.d[3]) + i] = rgb.data[i * 3 + c] / 255.0f;
      }
    }
    cudaMemcpyAsync(buffers_[0], host_input.data(), vol * 4, cudaMemcpyHostToDevice, stream_);
  }
}

void Yolo26nSeg::postprocess_mask_cpu(const cv::Mat& bgr_image, uint8_t* d_mask_canvas) {
  // Download results to CPU
  cudaMemcpyAsync(host_output0_raw_.data(), buffers_[1], host_output0_raw_.size(), cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(host_output1_raw_.data(), buffers_[2], host_output1_raw_.size(), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  cv::Mat mask_canvas = cv::Mat::zeros(bgr_image.size(), CV_8U);
  int num_dets = 128; // Matching MAX_DETECTIONS_SHARED
  int proto_h = 160, proto_w = 160, proto_c = 32;

  for (int i = 0; i < num_dets; ++i) {
    float score, x0, y0, x1, y1;
    int class_id;
    float coeffs[32];

    if (is_fp16_) {
      half* row = (half*)host_output0_raw_.data() + (i * 38);
      score = (float)row[4];
      if (score < conf_threshold_) continue;
      x0 = (float)row[0] * bgr_image.cols / 640.0f;
      y0 = (float)row[1] * bgr_image.rows / 640.0f;
      x1 = (float)row[2] * bgr_image.cols / 640.0f;
      y1 = (float)row[3] * bgr_image.rows / 640.0f;
      class_id = (int)row[5];
      for (int c = 0; c < 32; ++c) coeffs[c] = (float)row[6 + c];
    } else {
      float* row = (float*)host_output0_raw_.data() + (i * 38);
      score = row[4];
      if (score < conf_threshold_) continue;
      x0 = row[0] * bgr_image.cols / 640.0f;
      y0 = row[1] * bgr_image.rows / 640.0f;
      x1 = row[2] * bgr_image.cols / 640.0f;
      y1 = row[3] * bgr_image.rows / 640.0f;
      class_id = (int)row[5];
      for (int c = 0; c < 32; ++c) coeffs[c] = row[6 + c];
    }

    // Process mask for this detection
    for (int y = std::max(0, (int)y0); y < std::min(bgr_image.rows, (int)y1); ++y) {
      for (int x = std::max(0, (int)x0); x < std::min(bgr_image.cols, (int)x1); ++x) {
        int px = x * proto_w / bgr_image.cols;
        int py = y * proto_h / bgr_image.rows;
        float sum = 0.0f;
        for (int c = 0; c < 32; ++c) {
          float p_val;
          if (is_fp16_) p_val = (float)((half*)host_output1_raw_.data())[c * proto_w * proto_h + py * proto_w + px];
          else p_val = ((float*)host_output1_raw_.data())[c * proto_w * proto_h + py * proto_w + px];
          sum += coeffs[c] * p_val;
        }
        if (sigmoid(sum) > 0.5f) {
            mask_canvas.at<uint8_t>(y, x) = class_id + 1;
        }
      }
    }
  }
  cudaMemcpyAsync(d_mask_canvas, mask_canvas.data, bgr_image.total(), cudaMemcpyHostToDevice, stream_);
}

void Yolo26nSeg::infer_to_canvas(const cv::Mat& bgr_image, uint8_t* d_mask_canvas) {
  if (use_cuda_kernels_) {
    size_t src_size = bgr_image.total() * bgr_image.elemSize();
    cudaMemcpyAsync(d_src_image_, bgr_image.data, src_size, cudaMemcpyHostToDevice, stream_);

    // 1. Bilinear Preprocessing (Resizing + Normalization)
    launch_preprocess((uint8_t*)d_src_image_, buffers_[0], bgr_image.cols, bgr_image.rows, input_dims_.d[3], input_dims_.d[2], bgr_image.channels(), is_fp16_, stream_);
    
    // 2. TensorRT Inference Execution
    context_->setTensorAddress(input_name_.c_str(), buffers_[0]);
    context_->setTensorAddress(output0_name_.c_str(), buffers_[1]);
    context_->setTensorAddress(output1_name_.c_str(), buffers_[2]);
    context_->enqueueV3(stream_);

    // 3. Prototype Reformatting (CHW to HWC for cache locality)
    launch_reformat_prototypes(buffers_[2], d_proto_reformatted_, 160, 160, 32, is_fp16_, stream_);
    
    // 4. Semantic Mask Generation via GPU Kernel
    launch_postprocess_mask(buffers_[1], d_proto_reformatted_, d_mask_canvas, bgr_image.cols, bgr_image.rows, conf_threshold_, is_fp16_, stream_);
  } else {
    // STANDARD BASELINE PATH (Standard GPU Inference + CPU Support)
    preprocess_cpu(bgr_image);
    
    context_->setTensorAddress(input_name_.c_str(), buffers_[0]);
    context_->setTensorAddress(output0_name_.c_str(), buffers_[1]);
    context_->setTensorAddress(output1_name_.c_str(), buffers_[2]);
    context_->enqueueV3(stream_);

    postprocess_mask_cpu(bgr_image, d_mask_canvas);
  }
}

std::vector<DetectedCone> Yolo26nSeg::infer(const cv::Mat &bgr_image) {
  // Trigger asynchronous download of the primary detection output
  cudaMemcpyAsync(host_output0_raw_.data(), buffers_[1], host_output0_raw_.size(), cudaMemcpyDeviceToHost, stream_);
  return postprocess(host_output0_raw_.data(), nullptr, bgr_image.size());
}

std::vector<DetectedCone> Yolo26nSeg::postprocess(void* outA, void*, const cv::Size &original_size) {
  std::vector<DetectedCone> detections;
  const int num_detections = 300;
  const int num_channels = 38;
  float scale_w = (float)original_size.width / 640.0f;
  float scale_h = (float)original_size.height / 640.0f;

  for (int i = 0; i < num_detections; ++i) {
    float score;
    float row_data[6];
    if (is_fp16_) {
      half* row = (half*)outA + (i * num_channels);
      score = (float)row[4];
      if (score < conf_threshold_) continue;
      for (int j=0; j<6; ++j) row_data[j] = (float)row[j];
    } else {
      float* row = (float*)outA + (i * num_channels);
      score = row[4];
      if (score < conf_threshold_) continue;
      for (int j=0; j<6; ++j) row_data[j] = row[j];
    }

    DetectedCone det;
    det.class_id = (int)row_data[5];
    det.yolo_confidence = score;
    // Map bounding box center back to original image coordinates
    det.center_2d = cv::Point2f(row_data[0] * scale_w + (row_data[2] * scale_w - row_data[0] * scale_w)/2.0f, 
                                row_data[1] * scale_h + (row_data[3] * scale_h - row_data[1] * scale_h)/2.0f);
    detections.push_back(det);
  }
  return detections;
}
