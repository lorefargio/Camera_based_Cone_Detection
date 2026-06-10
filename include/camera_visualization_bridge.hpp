#pragma once

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <opencv2/opencv.hpp>

#include "detected_cone.hpp"
#include "camera_pipeline_config.hpp"

namespace camera_perception {

/**
 * @class CameraVisualizationBridge
 * @brief Offloads OpenCV image drawing and ROS topic publication to a background thread.
 */
class CameraVisualizationBridge {
public:
    explicit CameraVisualizationBridge(rclcpp::Node* node);
    ~CameraVisualizationBridge();

    /**
     * @brief Enqueues image and detection snapshots for asynchronous visual processing.
     */
    void enqueue(
        const std_msgs::msg::Header& header,
        const cv::Mat& raw_image,
        const std::vector<uint8_t>& mask_data,
        const std::vector<DetectedCone>& detections,
        int height,
        int width,
        const CameraPipelineConfig& config);

    /**
     * @brief Terminates the worker loop.
     */
    void stop();

private:
    void workerLoop();

    struct CameraVizData {
        std_msgs::msg::Header header;
        cv::Mat raw_image;
        std::vector<uint8_t> mask_data;
        std::vector<DetectedCone> detections;
        int height;
        int width;
    };

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_canvas_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;

    std::deque<std::unique_ptr<CameraVizData>> queue_;
    std::thread worker_thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_thread_ = false;
    bool publish_debug_ = false;
};

} // namespace camera_perception
