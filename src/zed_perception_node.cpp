#include "zed_fusion_perception/zed_perception_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

ZedPerceptionNode::ZedPerceptionNode(const rclcpp::NodeOptions& options)
    : Node("zed_perception_node", options) {
    
    std::string engine_path = this->declare_parameter("engine_path", "models/yolo26n-seg.engine");
    float conf_threshold = this->declare_parameter("conf_threshold", 0.5);
    float nms_threshold = this->declare_parameter("nms_threshold", 0.45);
    publish_debug_ = this->declare_parameter("publish_debug", false);

    yolo_ = std::make_unique<Yolo26nSeg>(engine_path, conf_threshold, nms_threshold);

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/rgb/color/rect/image", 10, std::bind(&ZedPerceptionNode::imageCallback, this, std::placeholders::_1));

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/zed/zed_node/rgb/color/rect/camera_info", 10, std::bind(&ZedPerceptionNode::cameraInfoCallback, this, std::placeholders::_1));

    debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/debug_image", 10);
    debug_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/debug_mask_canvas", 10);
    mask_canvas_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/camera_mask_canvas", 10);
    camera_cones_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/perception/camera_cones", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/perception/markers", 10);
}

void ZedPerceptionNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    RCLCPP_INFO_ONCE(this->get_logger(), "Camera Info received!");
    camera_info_ = *msg;
}

void ZedPerceptionNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    auto start_node = std::chrono::high_resolution_clock::now();
    
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

    static uint8_t* d_mask_canvas = nullptr;
    static size_t canvas_size = 0;
    if (d_mask_canvas == nullptr || canvas_size != cv_ptr->image.total()) {
        if (d_mask_canvas) cudaFree(d_mask_canvas);
        canvas_size = cv_ptr->image.total();
        cudaMalloc(reinterpret_cast<void**>(&d_mask_canvas), canvas_size);
    }

    auto detections = yolo_->infer(cv_ptr->image);
    yolo_->infer_to_canvas(cv_ptr->image, d_mask_canvas);

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

    auto mask_msg = std::make_unique<sensor_msgs::msg::Image>();
    mask_msg->header = msg->header;
    mask_msg->height = cv_ptr->image.rows;
    mask_msg->width = cv_ptr->image.cols;
    mask_msg->encoding = "mono8";
    mask_msg->step = mask_msg->width;
    mask_msg->data.resize(canvas_size);
    cudaMemcpy(mask_msg->data.data(), d_mask_canvas, canvas_size, cudaMemcpyDeviceToHost);
    
    // Publichiamo la maschera RAW per il nodo di Fusion
    mask_canvas_pub_->publish(*mask_msg);

    // 4. Debug Image & Debug Mask
    if (publish_debug_) {
        // Pubblica Debug Image (Centri coni)
        if (debug_pub_->get_subscription_count() > 0) {
            cv::Mat debug_img = cv_ptr->image.clone();
            for (const auto& det : detections) {
                cv::Scalar color(0, 255, 0);
                if (det.class_id == 0) color = cv::Scalar(255, 0, 0);
                else if (det.class_id == 1) color = cv::Scalar(0, 255, 255);
                else if (det.class_id == 2) color = cv::Scalar(0, 165, 255);
                cv::circle(debug_img, det.center_2d, 5, color, -1);
            }
            debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", debug_img).toImageMsg());
        }

        // Pubblica Debug Mask (Colorata per occhio umano)
        if (debug_mask_pub_->get_subscription_count() > 0) {
            cv::Mat raw_mask(mask_msg->height, mask_msg->width, CV_8U, mask_msg->data.data());
            cv::Mat color_mask;
            // Scaliamo i valori (es: ID 1 diventa 50, ID 2 diventa 100...) e applichiamo mappa colori
            raw_mask.convertTo(color_mask, CV_8U, 25); 
            cv::applyColorMap(color_mask, color_mask, cv::COLORMAP_JET);
            // Forza a nero dove il valore originale era 0 (background)
            color_mask.setTo(cv::Scalar(0, 0, 0), raw_mask == 0);
            debug_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", color_mask).toImageMsg());
        }
    }

    auto end_node = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_node - start_node).count();
    double hz = 1000.0 / total_ms;

    RCLCPP_INFO(this->get_logger(), "LATENCY: %.2f ms | FREQUENCY: %.2f Hz | CONES: %zu", 
                total_ms, hz, detections.size());
}

geometry_msgs::msg::Point ZedPerceptionNode::projectTo3D(const cv::Point2f& center_2d, double depth) {
    geometry_msgs::msg::Point p;
    double fx = camera_info_.k[0];
    double cx = camera_info_.k[2];
    double fy = camera_info_.k[4];
    double cy = camera_info_.k[5];
    p.z = depth;
    p.x = (center_2d.x - cx) * depth / fx;
    p.y = (center_2d.y - cy) * depth / fy;
    return p;
}
