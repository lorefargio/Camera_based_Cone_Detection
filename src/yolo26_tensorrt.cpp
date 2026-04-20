#include "zed_fusion_perception/yolo26_tensorrt.hpp"
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

Yolo26nSeg::Yolo26nSeg(const std::string& engine_path, float conf_threshold, float nms_threshold)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
    loadEngine(engine_path);
    allocateBuffers();
    cudaStreamCreate(&stream_);
}

Yolo26nSeg::~Yolo26nSeg() {
    cudaStreamDestroy(stream_);
    for (void* buf : buffers_) {
        cudaFree(buf);
    }
    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
}

void Yolo26nSeg::loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    char* data = new char[size];
    file.read(data, size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(data, size);
    delete[] data;

    context_ = engine_->createExecutionContext();

    input_dims_ = engine_->getBindingDimensions(0);
    output0_dims_ = engine_->getBindingDimensions(1);
    output1_dims_ = engine_->getBindingDimensions(2);
}

void Yolo26nSeg::allocateBuffers() {
    size_t input_size = 1;
    for (int i = 0; i < input_dims_.nbDims; ++i) input_size *= input_dims_.d[i];
    cudaMalloc(&buffers_[0], input_size * sizeof(float));

    size_t output0_size = 1;
    for (int i = 0; i < output0_dims_.nbDims; ++i) output0_size *= output0_dims_.d[i];
    cudaMalloc(&buffers_[1], output0_size * sizeof(float));

    size_t output1_size = 1;
    for (int i = 0; i < output1_dims_.nbDims; ++i) output1_size *= output1_dims_.d[i];
    cudaMalloc(&buffers_[2], output1_size * sizeof(float));
}

void Yolo26nSeg::preprocess(const cv::Mat& src, float* dst) {
    int input_w = input_dims_.d[3];
    int input_h = input_dims_.d[2];
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(input_w, input_h));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int i = 0; i < 3; ++i) {
        memcpy(dst + i * input_w * input_h, channels[i].data, input_w * input_h * sizeof(float));
    }
}

std::vector<DetectedCone> Yolo26nSeg::infer(const cv::Mat& bgr_image) {
    size_t input_vol = 1;
    for (int i = 0; i < input_dims_.nbDims; ++i) input_vol *= input_dims_.d[i];
    std::vector<float> input_data(input_vol);
    preprocess(bgr_image, input_data.data());

    cudaMemcpyAsync(buffers_[0], input_data.data(), input_vol * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->enqueueV2(buffers_, stream_, nullptr);

    size_t out0_vol = 1;
    for (int i = 0; i < output0_dims_.nbDims; ++i) out0_vol *= output0_dims_.d[i];
    std::vector<float> out0_host(out0_vol);
    cudaMemcpyAsync(out0_host.data(), buffers_[1], out0_vol * sizeof(float), cudaMemcpyDeviceToHost, stream_);

    size_t out1_vol = 1;
    for (int i = 0; i < output1_dims_.nbDims; ++i) out1_vol *= output1_dims_.d[i];
    std::vector<float> out1_host(out1_vol);
    cudaMemcpyAsync(out1_host.data(), buffers_[2], out1_vol * sizeof(float), cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);

    return postprocess(out0_host.data(), out1_host.data(), bgr_image.size());
}

std::vector<DetectedCone> Yolo26nSeg::postprocess(float* output0, float* output1, const cv::Size& original_size) {
    std::vector<DetectedCone> detections;
    int num_anchors = output0_dims_.d[2];
    int data_size = output0_dims_.d[1]; // 4 + num_classes + 32
    int num_classes = data_size - 4 - 32;

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> mask_coeffs;

    float scale_w = (float)original_size.width / input_dims_.d[3];
    float scale_h = (float)original_size.height / input_dims_.d[2];

    for (int i = 0; i < num_anchors; ++i) {
        float* data = output0 + i; // Assuming [1, 39, 8400] stored as column-major or similar? 
        // Wait, if it's [1, 39, 8400], it's likely row-major in memory: [39][8400]
        // So anchor i data is at offset i, i + 8400, i + 2*8400, ...
        
        float x = output0[i] * scale_w;
        float y = output0[i + num_anchors] * scale_h;
        float w = output0[i + 2 * num_anchors] * scale_w;
        float h = output0[i + 3 * num_anchors] * scale_h;

        float max_score = 0;
        int class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = output0[i + (4 + c) * num_anchors];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score > conf_threshold_) {
            bboxes.push_back(cv::Rect(x - w / 2, y - h / 2, w, h));
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
            
            std::vector<float> coeffs(32);
            for (int j = 0; j < 32; ++j) {
                coeffs[j] = output0[i + (4 + num_classes + j) * num_anchors];
            }
            mask_coeffs.push_back(coeffs);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confidences, conf_threshold_, nms_threshold_, indices);

    cv::Mat prototypes(32, 160 * 160, CV_32F, output1);

    for (int idx : indices) {
        DetectedCone det;
        det.class_id = class_ids[idx];
        det.yolo_confidence = confidences[idx];
        det.center_2d = cv::Point2f(bboxes[idx].x + bboxes[idx].width / 2.0f, bboxes[idx].y + bboxes[idx].height / 2.0f);
        
        // Mask generation
        cv::Mat coeffs_mat(1, 32, CV_32F, mask_coeffs[idx].data());
        cv::Mat res = coeffs_mat * prototypes;
        cv::Mat mask_small(160, 160, CV_32F, res.data);
        
        for (int r = 0; r < 160; ++r) {
            for (int c = 0; c < 160; ++c) {
                mask_small.at<float>(r, c) = sigmoid(mask_small.at<float>(r, c));
            }
        }

        cv::Mat binary_mask;
        cv::threshold(mask_small, binary_mask, 0.5, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);

        // Resize mask back to original bounding box or full image?
        // Usually, we crop the mask to the bounding box region in the 160x160 space then resize.
        // For simplicity here, resize to full image and then crop or just keep full.
        // The blueprint says "binary mask" and "mask_area".
        
        cv::Mat mask_full;
        cv::resize(binary_mask, mask_full, original_size, 0, 0, cv::INTER_LINEAR);
        
        // Optional: crop mask_full to bounding box to avoid bleed
        cv::Mat cropped_mask = cv::Mat::zeros(original_size, CV_8U);
        cv::Rect roi = bboxes[idx] & cv::Rect(0, 0, original_size.width, original_size.height);
        if (roi.width > 0 && roi.height > 0) {
            mask_full(roi).copyTo(cropped_mask(roi));
        }

        det.mask = cropped_mask;
        det.mask_area = cv::countNonZero(det.mask);
        detections.push_back(det);
    }

    return detections;
}
