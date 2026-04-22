#include "zed_fusion_perception/zed_perception_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

ZedPerceptionNode::ZedPerceptionNode(const rclcpp::NodeOptions& options)
    : Node("zed_perception_node", options) {
    
    // Parameters
    std::string engine_path = this->declare_parameter("engine_path", "models/yolo26n-seg.engine");
    float conf_threshold = this->declare_parameter("conf_threshold", 0.5);
    float nms_threshold = this->declare_parameter("nms_threshold", 0.45);

    // Initialize YOLO
    yolo_ = std::make_unique<Yolo26nSeg>(engine_path, conf_threshold, nms_threshold);

    // Subscriptions
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>("/zed/zed_node/rgb/color/rect/image", 10, std::bind(&ZedPerceptionNode::imageCallback, this, std::placeholders::_1));

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>( "/zed2i/zed_node/rgb/camera_info", 10, std::bind(&ZedPerceptionNode::cameraInfoCallback, this, std::placeholders::_1));

    // Publishers
    debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/debug_image", 10);
    mask_canvas_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/camera_mask_canvas", 10);
    camera_cones_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/perception/camera_cones", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/perception/markers", 10);
}

void ZedPerceptionNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    camera_info_ = *msg;
}

void ZedPerceptionNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    if (camera_info_.k.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for camera info...");
        return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // 1. Run YOLO26n Inference
    auto start_yolo = std::chrono::high_resolution_clock::now();
    auto detections = yolo_->infer(cv_ptr->image);
    auto end_yolo = std::chrono::high_resolution_clock::now();
    
    // 2. Publish results (Camera only detections)
    geometry_msgs::msg::PoseArray camera_cones_msg;
    camera_cones_msg.header = msg->header;
    
    visualization_msgs::msg::MarkerArray markers;
    cv::Mat mask_canvas = cv::Mat::zeros(cv_ptr->image.size(), CV_8U);
    uint8_t cone_idx = 1;

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        geometry_msgs::msg::Pose pose;
        pose.position = projectTo3D(det.center_2d, 5.0); 
        camera_cones_msg.poses.push_back(pose);

        // Marker for visualization
        visualization_msgs::msg::Marker marker;
        marker.header = msg->header;
        marker.ns = "camera_cones";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position = pose.position;
        marker.scale.x = 0.228; // standard cone diameter
        marker.scale.y = 0.228;
        marker.scale.z = 0.325; // standard cone height
        
        // Color based on class_id (0: blue, 1: yellow, 2: orange)
        if (det.class_id == 0) { marker.color.b = 1.0; marker.color.a = 1.0; }
        else if (det.class_id == 1) { marker.color.r = 1.0; marker.color.g = 1.0; marker.color.a = 1.0; }
        else if (det.class_id == 2) { marker.color.r = 1.0; marker.color.g = 0.5; marker.color.a = 1.0; }
        else { marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0; }

        markers.markers.push_back(marker);

        // Fill canvas with index
        if (!det.mask.empty()) {
            mask_canvas.setTo(cone_idx, det.mask);
        }
        if (cone_idx < 255) cone_idx++;
    }
    camera_cones_pub_->publish(camera_cones_msg);
    marker_pub_->publish(markers);

    // Publish Mask Canvas
    auto mask_msg = cv_bridge::CvImage(msg->header, "mono8", mask_canvas).toImageMsg();
    mask_canvas_pub_->publish(*mask_msg);

    auto end_total = std::chrono::high_resolution_clock::now();
    
    double yolo_ms = std::chrono::duration<double, std::milli>(end_yolo - start_yolo).count();
    double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    
    if (debug_pub_->get_subscription_count() > 0) {
        cv::Mat debug_img = cv_ptr->image.clone();
        for (const auto& det : detections) {
            cv::Scalar color(0, 255, 0);
            if (det.class_id == 0) color = cv::Scalar(255, 0, 0); // Blue
            else if (det.class_id == 1) color = cv::Scalar(0, 255, 255); // Yellow
            else if (det.class_id == 2) color = cv::Scalar(0, 165, 255); // Orange

            cv::circle(debug_img, det.center_2d, 5, color, -1);
            cv::putText(debug_img, std::to_string(det.final_confidence), det.center_2d, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            if (!det.mask.empty()) {
                // Draw mask overlay
                cv::Mat mask_bgr;
                cv::cvtColor(det.mask, mask_bgr, cv::COLOR_GRAY2BGR);
                // Colorize mask
                mask_bgr.setTo(color, det.mask);
                cv::addWeighted(debug_img, 1.0, mask_bgr, 0.4, 0, debug_img);
            }
        }
        auto debug_msg = cv_bridge::CvImage(msg->header, "bgr8", debug_img).toImageMsg();
        debug_pub_->publish(*debug_msg);
    }
}

geometry_msgs::msg::Point ZedPerceptionNode::projectTo3D(const cv::Point2f& center_2d, double depth) {
    geometry_msgs::msg::Point p;
    // Pinhole camera model
    // x = (u - cx) * z / fx
    // y = (v - cy) * z / fy
    double fx = camera_info_.k[0];
    double cx = camera_info_.k[2];
    double fy = camera_info_.k[4];
    double cy = camera_info_.k[5];

    p.z = depth;
    p.x = (center_2d.x - cx) * depth / fx;
    p.y = (center_2d.y - cy) * depth / fy;

    return p;
}
