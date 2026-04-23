#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "zed_fusion_perception/yolo26_tensorrt.hpp"

class ZedPerceptionNode : public rclcpp::Node {
public:
    ZedPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_mask_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_canvas_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr camera_cones_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    
    std::unique_ptr<Yolo26nSeg> yolo_;
    sensor_msgs::msg::CameraInfo camera_info_;
    bool publish_debug_;
    
    geometry_msgs::msg::Point projectTo3D(const cv::Point2f& center_2d, double depth);
};
