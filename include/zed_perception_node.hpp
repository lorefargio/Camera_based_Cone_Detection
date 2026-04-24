#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <fstream>
#include "yolo26_tensorrt.hpp"

/**
 * @brief ROS 2 Node for camera-based cone perception.
 * 
 * This node subscribes to raw ZED images and publishes semantic segmentation masks.
 * It is implemented as a ROS 2 Component to support Zero-Copy IPC.
 */
class ZedPerceptionNode : public rclcpp::Node {
public:
    /**
     * @brief Constructor for ZedPerceptionNode.
     * @param options Node options for ROS 2 configuration.
     */
    ZedPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    /**
     * @brief Callback for incoming image messages. Executes the perception pipeline.
     * @param msg Shared pointer to the incoming sensor_msgs::msg::Image.
     */
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    /**
     * @brief Callback for camera calibration information.
     * @param msg Shared pointer to the incoming sensor_msgs::msg::CameraInfo.
     */
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // ROS 2 Communication
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_canvas_pub_;
    
    // Core Logic
    std::unique_ptr<Yolo26nSeg> yolo_;
    sensor_msgs::msg::CameraInfo camera_info_;
    
    // Parameters and Statistics
    bool publish_debug_;
    bool export_stats_;
    std::ofstream stats_file_;
};
