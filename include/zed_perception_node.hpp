#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <fstream>
#include <memory>

#include "camera_pipeline_config.hpp"
#include "camera_perception_pipeline.hpp"
#include "camera_visualization_bridge.hpp"

/**
 * @class ZedPerceptionNode
 * @brief Minimal ROS 2 wrapper for camera-based cone perception.
 */
class ZedPerceptionNode : public rclcpp::Node {
public:
    explicit ZedPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~ZedPerceptionNode() override;

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void loadParametersToConfig();

    // ROS 2 Communication
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    
    // Core Components
    camera_perception::CameraPipelineConfig config_;
    std::unique_ptr<camera_perception::CameraPerceptionPipeline> pipeline_;
    std::unique_ptr<camera_perception::CameraVisualizationBridge> viz_bridge_;
    
    sensor_msgs::msg::CameraInfo camera_info_;
    std::ofstream stats_file_;
    int iter_count_ = 0;
};
