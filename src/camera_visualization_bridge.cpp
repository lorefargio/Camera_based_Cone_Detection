#include "camera_visualization_bridge.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>

namespace camera_perception {

CameraVisualizationBridge::CameraVisualizationBridge(rclcpp::Node* node) {
    debug_pub_ = node->create_publisher<sensor_msgs::msg::Image>("/perception/debug_image", 10);
    debug_mask_pub_ = node->create_publisher<sensor_msgs::msg::Image>("/perception/debug_mask_canvas", 10);
    mask_canvas_pub_ = node->create_publisher<sensor_msgs::msg::Image>("/perception/camera_mask_canvas", 10);

    worker_thread_ = std::thread(&CameraVisualizationBridge::workerLoop, this);
}

CameraVisualizationBridge::~CameraVisualizationBridge() {
    stop();
}

void CameraVisualizationBridge::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_thread_) return;
        stop_thread_ = true;
    }
    cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void CameraVisualizationBridge::enqueue(
    const std_msgs::msg::Header& header,
    const cv::Mat& raw_image,
    const std::vector<uint8_t>& mask_data,
    const std::vector<DetectedCone>& detections,
    int height,
    int width,
    const CameraPipelineConfig& config) {

    std::lock_guard<std::mutex> lock(mutex_);
    publish_debug_ = config.publish_debug;

    auto data = std::make_unique<CameraVizData>();
    data->header = header;
    data->mask_data = mask_data;
    data->detections = detections;
    data->height = height;
    data->width = width;

    if (publish_debug_ && !raw_image.empty()) {
        data->raw_image = raw_image.clone();
    }

    queue_.push_back(std::move(data));
    if (queue_.size() > 50) {
        queue_.pop_front();
    }
    cv_.notify_one();
}

void CameraVisualizationBridge::workerLoop() {
    auto get_class_color = [](int class_id) {
        if (class_id == 0) return cv::Scalar(255, 0, 0);      // Blue
        if (class_id == 1) return cv::Scalar(128, 128, 128);  // Fallen (Gray)
        if (class_id == 2) return cv::Scalar(0, 165, 255);    // Orange
        if (class_id == 3) return cv::Scalar(0, 0, 255);      // Big Orange (Red)
        if (class_id == 4) return cv::Scalar(0, 255, 255);    // Yellow
        return cv::Scalar(255, 255, 255);
    };

    while (true) {
        std::unique_ptr<CameraVizData> data;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || stop_thread_; });

            if (stop_thread_ && queue_.empty()) break;

            if (!queue_.empty()) {
                data = std::move(queue_.front());
                queue_.pop_front();
            }
        }

        if (!data) continue;

        // 1. Build and Publish Main Mask Canvas
        auto mask_msg = std::make_unique<sensor_msgs::msg::Image>();
        mask_msg->header = data->header;
        mask_msg->height = data->height;
        mask_msg->width = data->width;
        mask_msg->encoding = "mono8";
        mask_msg->step = mask_msg->width;
        mask_msg->data = std::move(data->mask_data);

        // 3. Debug Visualizations
        if (publish_debug_) {
            cv::Mat raw_mask(data->height, data->width, CV_8U, (void*)mask_msg->data.data());

            if (debug_pub_->get_subscription_count() > 0 && !data->raw_image.empty()) {
                cv::Mat debug_img = data->raw_image;
                for (const auto& det : data->detections) {
                    cv::circle(debug_img, det.center_2d, 5, get_class_color(det.class_id), -1);
                }
                debug_pub_->publish(*cv_bridge::CvImage(data->header, "bgr8", debug_img).toImageMsg());
            }

            if (debug_mask_pub_->get_subscription_count() > 0) {
                cv::Mat color_mask = cv::Mat::zeros(raw_mask.size(), CV_8UC3);
                for (int class_id = 0; class_id < 5; ++class_id) {
                    color_mask.setTo(get_class_color(class_id), raw_mask == (class_id + 1));
                }
                debug_mask_pub_->publish(*cv_bridge::CvImage(data->header, "bgr8", color_mask).toImageMsg());
            }
        }

        mask_canvas_pub_->publish(std::move(mask_msg));
    }
}

} // namespace camera_perception
