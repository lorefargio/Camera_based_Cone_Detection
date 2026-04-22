#include "zed_fusion_perception/yolo26_tensorrt.hpp"
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>

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
  for (void *buf : buffers_)
    cudaFree(buf);
}

void Yolo26nSeg::loadEngine(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good())
    throw std::runtime_error("Failed to read engine file: " + path);

  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> data(size);
  file.read(data.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(gLogger));
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));

  // CREIAMO IL CONTEXT PRIMA DI USARLO
  context_.reset(engine_->createExecutionContext());

  // In TRT 10 gli indici possono variare. Identifichiamo i tensori per NOME e
  // SHAPE
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    std::string name = engine_->getIOTensorName(i);
    nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
    nvinfer1::DataType type = engine_->getTensorDataType(name.c_str());

    // Stampa diagnostica fondamentale
    std::cout << "Tensor " << i << ": " << name << " | Shape: ";
    for (int j = 0; j < dims.nbDims; ++j)
      std::cout << dims.d[j] << " ";
    std::cout << "| DataType: " << (int)type << " (0=FP32, 1=FP16)"
              << std::endl;

    // Assegnazione dinamica basata sulle dimensioni
    if (dims.nbDims == 4 && dims.d[1] == 3) {
      input_name_ = name;
      input_dims_ = dims;
    } else if (dims.nbDims == 3) {
      output0_name_ = name; // Box + Classi (es. 1x41x8400)
      output0_dims_ = dims;
    } else if (dims.nbDims == 4 && dims.d[1] == 32) {
      output1_name_ = name; // Prototypes (es. 1x32x160x160)
      output1_dims_ = dims;
    }
  }
}

void Yolo26nSeg::allocateBuffers() {
  auto get_size = [](nvinfer1::Dims dims) {
    size_t s = 1;
    for (int i = 0; i < dims.nbDims; ++i)
      s *= dims.d[i];
    return s;
  };

  size_t in_s = get_size(input_dims_);
  size_t out0_s = get_size(output0_dims_);
  size_t out1_s = get_size(output1_dims_);

  // Allocazione GPU
  cudaMalloc(&buffers_[0], in_s * sizeof(float));
  cudaMalloc(&buffers_[1], out0_s * sizeof(float));
  cudaMalloc(&buffers_[2], out1_s * sizeof(float));

  // Allocazione HOST PERSISTENTE (Previene corruzione asincrona)
  host_input_batch_.resize(in_s);
  host_output0_.resize(out0_s);
  host_output1_.resize(out1_s);
}

void Yolo26nSeg::preprocess(const cv::Mat &src, float *dst) {
  cv::Mat bgr;
  if (src.channels() == 4)
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
  else
    bgr = src;

  cv::Size input_size(input_dims_.d[3], input_dims_.d[2]);

  // Ricalca il preprocess dello script funzionante
  cv::Mat blob;
  cv::dnn::blobFromImage(bgr, blob, 1.0 / 255.0, input_size,
                         cv::Scalar(0, 0, 0), true, false, CV_32F);


  memcpy(dst, blob.ptr<float>(0), blob.total() * sizeof(float));
}

std::vector<DetectedCone> Yolo26nSeg::infer(const cv::Mat &bgr_image) {
  preprocess(bgr_image, host_input_batch_.data());

  // HtoD
  cudaMemcpyAsync(buffers_[0], host_input_batch_.data(),
                  host_input_batch_.size() * sizeof(float),
                  cudaMemcpyHostToDevice, stream_);

  // TRT 10 Binding
  context_->setTensorAddress(input_name_.c_str(), buffers_[0]);
  context_->setTensorAddress(output0_name_.c_str(), buffers_[1]);
  context_->setTensorAddress(output1_name_.c_str(), buffers_[2]);

  // Esecuzione
  if (!context_->enqueueV3(stream_))
    return {};

  // DtoH
  cudaMemcpyAsync(host_output0_.data(), buffers_[1],
                  host_output0_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                  stream_);
  cudaMemcpyAsync(host_output1_.data(), buffers_[2],
                  host_output1_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                  stream_);

  // SINCRONIZZAZIONE (Cruciale per evitare quadratini scattanti)
  cudaStreamSynchronize(stream_);

  return postprocess(host_output0_.data(), host_output1_.data(),
                     bgr_image.size());
}

std::vector<DetectedCone>
Yolo26nSeg::postprocess(float *outA, float *outB,
                        const cv::Size &original_size) {
  std::vector<DetectedCone> detections;

  // Parametri estratti dai tuoi log
  const int num_detections = 300;
  const int num_channels = 38;

  // Rapporto di scala (il modello lavora su 640x640)
  float scale_w = (float)original_size.width / 640.0f;
  float scale_h = (float)original_size.height / 640.0f;

  for (int i = 0; i < num_detections; ++i) {
    float *row = outA + (i * num_channels);

    float score = row[4]; // La confidenza è già al canale 4

    // Se lo score è 0, significa che abbiamo finito i rilevamenti validi
    // (i modelli end2end riempiono il resto con zeri)
    if (score < conf_threshold_)
      continue;

    DetectedCone det;
    det.class_id = (int)row[5]; // Il Class ID è al canale 5
    det.yolo_confidence = score;

    // Coordinate x1, y1, x2, y2 (confermate dai tuoi log: 130, 449...)
    float x1 = row[0] * scale_w;
    float y1 = row[1] * scale_h;
    float x2 = row[2] * scale_w;
    float y2 = row[3] * scale_h;

    // Calcoliamo centro e box per OpenCV
    float w = x2 - x1;
    float h = y2 - y1;
    cv::Rect bbox(x1, y1, w, h);
    det.center_2d = cv::Point2f(x1 + w / 2.0f, y1 + h / 2.0f);

    // --- Elaborazione Maschera ---
    // I prototipi sono sempre nel secondo buffer (outB)
    cv::Mat protos(32, 160 * 160, CV_32F, outB);

    // I coefficienti partono dal canale 6 e sono 32
    cv::Mat coeffs(1, 32, CV_32F, row + 6);

    cv::Mat res = coeffs * protos;
    cv::Mat mask_small(160, 160, CV_32F, res.data);

    // Sigmoide e soglia per la maschera
    cv::Mat binary_mask;
    cv::exp(-mask_small, binary_mask);
    binary_mask = 1.0 / (1.0 + binary_mask);
    cv::threshold(binary_mask, binary_mask, 0.5, 255, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8U);

    // Resize e pulizia ROI
    cv::resize(binary_mask, binary_mask, original_size);
    cv::Mat final_mask = cv::Mat::zeros(original_size, CV_8U);
    cv::Rect roi =
        bbox & cv::Rect(0, 0, original_size.width, original_size.height);
    if (roi.width > 0 && roi.height > 0) {
      binary_mask(roi).copyTo(final_mask(roi));
    }

    det.mask = final_mask;
    det.mask_area = cv::countNonZero(det.mask);

    detections.push_back(det);
  }

  return detections;
}