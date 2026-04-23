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
        // Use toCvShare to avoid copying if possible (Intra-process)
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Allocate GPU buffer for mask canvas if needed
    static uint8_t* d_mask_canvas = nullptr;
    static size_t canvas_size = 0;
    size_t current_size = cv_ptr->image.total();
    if (d_mask_canvas == nullptr || canvas_size != current_size) {
        if (d_mask_canvas) cudaFree(d_mask_canvas);
        canvas_size = current_size;
        cudaMalloc(&d_mask_canvas, canvas_size);
    }

    // 1. Run YOLO26n High-Performance Inference (directly to GPU canvas)
    auto start_yolo = std::chrono::high_resolution_clock::now();
    yolo_->infer_to_canvas(cv_ptr->image, d_mask_canvas);
    
    // Also get detections for other publishers (Markers, etc.)
    // Note: This could be optimized further if Markers aren't always needed
    auto detections = yolo_->infer(cv_ptr->image); 
    auto end_yolo = std::chrono::high_resolution_clock::now();
    
    // 2. Publish results
    geometry_msgs::msg::PoseArray camera_cones_msg;
    camera_cones_msg.header = msg->header;
    visualization_msgs::msg::MarkerArray markers;

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        geometry_msgs::msg::Pose pose;
        pose.position = projectTo3D(det.center_2d, 5.0); 
        camera_cones_msg.poses.push_back(pose);

        visualization_msgs::msg::Marker marker;
        marker.header = msg->header;
        marker.ns = "camera_cones";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.pose.position = pose.position;
        marker.scale.x = 0.228; marker.scale.y = 0.228; marker.scale.z = 0.325;
        
        if (det.class_id == 0) { marker.color.b = 1.0; marker.color.a = 1.0; }
        else if (det.class_id == 1) { marker.color.r = 1.0; marker.color.g = 1.0; marker.color.a = 1.0; }
        else if (det.class_id == 2) { marker.color.r = 1.0; marker.color.g = 0.5; marker.color.a = 1.0; }
        
        markers.markers.push_back(marker);
    }
    camera_cones_pub_->publish(camera_cones_msg);
    marker_pub_->publish(markers);

    // 3. Publish Mask Canvas (Copy from GPU to ROS Message)
    auto mask_msg = std::make_unique<sensor_msgs::msg::Image>();
    mask_msg->header = msg->header;
    mask_msg->height = cv_ptr->image.rows;
    mask_msg->width = cv_ptr->image.cols;
    mask_msg->encoding = "mono8";
    mask_msg->step = mask_msg->width;
    mask_msg->data.resize(canvas_size);
    cudaMemcpy(mask_msg->data.data(), d_mask_canvas, canvas_size, cudaMemcpyDeviceToHost);
    mask_canvas_pub_->publish(std::move(mask_msg));

    auto end_total = std::chrono::high_resolution_clock::now();
    double yolo_ms = std::chrono::duration<double, std::milli>(end_yolo - start_yolo).count();
    double total_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    
    RCLCPP_DEBUG(this->get_logger(), "Inference: %.2f ms, Total: %.2f ms", yolo_ms, total_ms);
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
