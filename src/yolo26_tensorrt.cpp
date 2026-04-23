#include "zed_fusion_perception/yolo26_tensorrt.hpp"
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << "[TRT] " << msg << std::endl;
  }
} gLogger;

Yolo26nSeg::Yolo26nSeg(const std::string &engine_path, float conf_threshold,
                       float nms_threshold)
    : graph_initialized_(false), conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
  loadEngine(engine_path);
  allocateBuffers();
  cudaStreamCreate(&stream_);
}

Yolo26nSeg::~Yolo26nSeg() {
  if (graph_initialized_) {
    cudaGraphExecDestroy(instance_);
    cudaGraphDestroy(graph_);
  }
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

  is_fp16_ = false;
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string name = engine_->getIOTensorName(i);
    nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
    nvinfer1::DataType type = engine_->getTensorDataType(name.c_str());
    if (type == nvinfer1::DataType::kHALF) is_fp16_ = true;
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
  cudaMalloc(&d_src_image_, 2208 * 1242 * 4); 
  cudaMalloc(&d_proto_reformatted_, get_size(output1_dims_) * element_size);
  host_output0_raw_.resize(get_size(output0_dims_) * element_size);
}

void Yolo26nSeg::initGraph(const cv::Mat& sample_img, uint8_t* d_mask_canvas) {
  context_->setTensorAddress(input_name_.c_str(), buffers_[0]);
  context_->setTensorAddress(output0_name_.c_str(), buffers_[1]);
  context_->setTensorAddress(output1_name_.c_str(), buffers_[2]);
  context_->enqueueV3(stream_);
  cudaStreamSynchronize(stream_);

  cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
  launch_preprocess((uint8_t*)d_src_image_, buffers_[0], sample_img.cols, sample_img.rows, input_dims_.d[3], input_dims_.d[2], sample_img.channels(), is_fp16_, stream_);
  context_->setTensorAddress(input_name_.c_str(), buffers_[0]);
  context_->setTensorAddress(output0_name_.c_str(), buffers_[1]);
  context_->setTensorAddress(output1_name_.c_str(), buffers_[2]);
  context_->enqueueV3(stream_);
  launch_reformat_prototypes(buffers_[2], d_proto_reformatted_, 160, 160, 32, is_fp16_, stream_);
  launch_postprocess_mask(buffers_[1], d_proto_reformatted_, d_mask_canvas, sample_img.cols, sample_img.rows, conf_threshold_, is_fp16_, stream_);
  cudaStreamEndCapture(stream_, &graph_);
  cudaGraphInstantiate(&instance_, graph_, 0);
  graph_initialized_ = true;
}

void Yolo26nSeg::infer_to_canvas(const cv::Mat& bgr_image, uint8_t* d_mask_canvas) {
  size_t src_size = bgr_image.total() * bgr_image.elemSize();
  cudaMemcpyAsync(d_src_image_, bgr_image.data, src_size, cudaMemcpyHostToDevice, stream_);

  if (!graph_initialized_) {
    initGraph(bgr_image, d_mask_canvas);
  }
  
  cudaGraphLaunch(instance_, stream_);
  // NO SYNC HERE: Node will handle it
}

std::vector<DetectedCone> Yolo26nSeg::infer(const cv::Mat &bgr_image) {
  // DtoH is async, no sync here
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
    det.center_2d = cv::Point2f(row_data[0] * scale_w + (row_data[2] * scale_w - row_data[0] * scale_w)/2.0f, 
                                row_data[1] * scale_h + (row_data[3] * scale_h - row_data[1] * scale_h)/2.0f);
    detections.push_back(det);
  }
  return detections;
}
