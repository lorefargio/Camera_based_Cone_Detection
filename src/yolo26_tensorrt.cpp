#include "zed_fusion_perception/yolo26_tensorrt.hpp"
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

// Logger TensorRT
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

Yolo26nSeg::Yolo26nSeg(const std::string &engine_path, float conf_threshold,
                       float nms_threshold)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
  loadEngine(engine_path);
  allocateBuffers();
  cudaStreamCreate(&stream_);
}

Yolo26nSeg::~Yolo26nSeg() {
  cudaStreamDestroy(stream_);
  for (void *buf : buffers_) cudaFree(buf);
  cudaFree(d_src_image_);
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

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string name = engine_->getIOTensorName(i);
    nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
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

  cudaMalloc(&buffers_[0], get_size(input_dims_) * sizeof(float));
  cudaMalloc(&buffers_[1], get_size(output0_dims_) * sizeof(float));
  cudaMalloc(&buffers_[2], get_size(output1_dims_) * sizeof(float));
  
  // Max source image size (e.g., 2K resolution)
  cudaMalloc(&d_src_image_, 2208 * 1242 * 4); 

  host_output0_.resize(get_size(output0_dims_));
  host_output1_.resize(get_size(output1_dims_));
}

void Yolo26nSeg::preprocess_gpu(const cv::Mat &src) {
  size_t src_size = src.total() * src.elemSize();
  cudaMemcpyAsync(d_src_image_, src.data, src_size, cudaMemcpyHostToDevice, stream_);
  
  launch_preprocess((uint8_t*)d_src_image_, (float*)buffers_[0], 
                    src.cols, src.rows, input_dims_.d[3], input_dims_.d[2], 
                    src.channels(), stream_);
}

std::vector<DetectedCone> Yolo26nSeg::infer(const cv::Mat &bgr_image) {
  preprocess_gpu(bgr_image);

  context_->setTensorAddress(input_name_.c_str(), buffers_[0]);
  context_->setTensorAddress(output0_name_.c_str(), buffers_[1]);
  context_->setTensorAddress(output1_name_.c_str(), buffers_[2]);

  if (!context_->enqueueV3(stream_)) return {};

  cudaMemcpyAsync(host_output0_.data(), buffers_[1], host_output0_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  cudaMemcpyAsync(host_output1_.data(), buffers_[2], host_output1_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  return postprocess(host_output0_.data(), host_output1_.data(), bgr_image.size());
}

void Yolo26nSeg::infer_to_canvas(const cv::Mat& bgr_image, uint8_t* d_mask_canvas) {
  // Nota: l'inferenza è già stata fatta da infer(). 
  // Qui lanciamo solo il kernel di post-processing sulla GPU.
  launch_postprocess_mask((float*)buffers_[1], (float*)buffers_[2], d_mask_canvas, 
                          bgr_image.cols, bgr_image.rows, conf_threshold_, stream_);
}

std::vector<DetectedCone> Yolo26nSeg::postprocess(float *outA, float *outB, const cv::Size &original_size) {
  std::vector<DetectedCone> detections;
  const int num_detections = 300;
  const int num_channels = 38;

  float scale_w = (float)original_size.width / 640.0f;
  float scale_h = (float)original_size.height / 640.0f;

  for (int i = 0; i < num_detections; ++i) {
    float *row = outA + (i * num_channels);
    float score = row[4];
    if (score < conf_threshold_) continue;

    DetectedCone det;
    det.class_id = (int)row[5];
    det.yolo_confidence = score;

    float x1 = row[0] * scale_w;
    float y1 = row[1] * scale_h;
    float x2 = row[2] * scale_w;
    float y2 = row[3] * scale_h;

    float w = x2 - x1;
    float h = y2 - y1;
    det.center_2d = cv::Point2f(x1 + w / 2.0f, y1 + h / 2.0f);

    // Note: CPU mask generation is kept here for debug/visualization, 
    // but the high-performance path uses infer_to_canvas.
    detections.push_back(det);
  }
  return detections;
}
