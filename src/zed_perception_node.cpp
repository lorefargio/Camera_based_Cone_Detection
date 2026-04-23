#include "zed_fusion_perception/zed_perception_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

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

    for (const auto& det : detections) {
        geometry_msgs::msg::Pose pose;
        pose.position = projectTo3D(det.center_2d, 5.0); 
        camera_cones_msg.poses.push_back(pose);
    }
    camera_cones_pub_->publish(camera_cones_msg);

    auto mask_msg = std::make_unique<sensor_msgs::msg::Image>();
    mask_msg->header = msg->header;
    mask_msg->height = cv_ptr->image.rows;
    mask_msg->width = cv_ptr->image.cols;
    mask_msg->encoding = "mono8";
    mask_msg->step = mask_msg->width;
    mask_msg->data.resize(canvas_size);
    cudaMemcpy(mask_msg->data.data(), d_mask_canvas, canvas_size, cudaMemcpyDeviceToHost);
    
    mask_canvas_pub_->publish(*mask_msg);

    if (publish_debug_) {
        cv::Mat raw_mask(mask_msg->height, mask_msg->width, CV_8U, mask_msg->data.data());
        
        auto get_class_color = [](int class_id) {
            if (class_id == 0) return cv::Scalar(255, 0, 0);   // Blue
            if (class_id == 1) return cv::Scalar(0, 255, 255); // Yellow
            if (class_id == 2) return cv::Scalar(0, 165, 255); // Orange
            if (class_id == 3) return cv::Scalar(0, 0, 255);   // Big Orange (Red)
            if (class_id == 4) return cv::Scalar(128, 128, 128); // Fallen (Gray)
            return cv::Scalar(255, 255, 255); // Default White
        };

        if (debug_pub_->get_subscription_count() > 0) {
            cv::Mat debug_img = cv_ptr->image.clone();
            for (const auto& det : detections) {
                cv::circle(debug_img, det.center_2d, 5, get_class_color(det.class_id), -1);
            }
            debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", debug_img).toImageMsg());
        }

        if (debug_mask_pub_->get_subscription_count() > 0) {
            cv::Mat color_mask = cv::Mat::zeros(raw_mask.size(), CV_8UC3);
            for (size_t i = 0; i < detections.size(); ++i) {
                uint8_t id = static_cast<uint8_t>(i + 1);
                color_mask.setTo(get_class_color(detections[i].class_id), raw_mask == id);
            }
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
