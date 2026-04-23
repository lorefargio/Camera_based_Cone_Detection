#include "zed_fusion_perception/zed_perception_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <fstream>

ZedPerceptionNode::ZedPerceptionNode(const rclcpp::NodeOptions& options)
    : Node("zed_perception_node", options) {
    
    std::string engine_path = this->declare_parameter("engine_path", "models/yolo26n-seg.engine");
    float conf_threshold = this->declare_parameter("conf_threshold", 0.5);
    float nms_threshold = this->declare_parameter("nms_threshold", 0.45);
    publish_debug_ = this->declare_parameter("publish_debug", false);
    export_stats_ = this->declare_parameter("export_stats", false);

    yolo_ = std::make_unique<Yolo26nSeg>(engine_path, conf_threshold, nms_threshold);

    RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");
    RCLCPP_INFO(this->get_logger(), " ZED PERCEPTION NODE INITIALIZED");
    RCLCPP_INFO(this->get_logger(), " - Model: YOLO26n-Seg (End-to-End)");
    RCLCPP_INFO(this->get_logger(), " - Inference: TensorRT 10.x + CUDA Graphs");
    RCLCPP_INFO(this->get_logger(), " - Optimization: Zero-Copy (toCvShare) Enabled");
    RCLCPP_INFO(this->get_logger(), " - Preprocessing: GPU Bilinear Resize");
    RCLCPP_INFO(this->get_logger(), " - Postprocessing: Shared Memory HWC Reformatting");
    RCLCPP_INFO(this->get_logger(), " - Stats Export: %s", export_stats_ ? "ENABLED" : "DISABLED");
    RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");

    if (export_stats_) {
        stats_file_.open("camera_stats.csv");
        stats_file_ << "timestamp,latency_ms,hz,detections\n";
    }

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

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        // ZERO-COPY: toCvShare avoids copying the buffer if it's already in the same process
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
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

    // --- CUDA GRAPH PIPELINE ---
    // infer_to_canvas runs the captured Graph (Preprocess -> Inference -> Reformat -> Mask)
    yolo_->infer_to_canvas(cv_ptr->image, d_mask_canvas);
    
    // Get detections on host for publishing (uses already computed data in GPU output buffer)
    auto detections = yolo_->infer(cv_ptr->image);

    geometry_msgs::msg::PoseArray camera_cones_msg;
    camera_cones_msg.header = msg->header;
    for (const auto& det : detections) {
        geometry_msgs::msg::Pose pose;
        pose.position = projectTo3D(det.center_2d, 5.0); 
        camera_cones_msg.poses.push_back(pose);
    }
    camera_cones_pub_->publish(camera_cones_msg);

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

    if (publish_debug_) {
        cv::Mat raw_mask(cv_ptr->image.rows, cv_ptr->image.cols, CV_8U, (void*)mask_msg->data.data());
        auto get_class_color = [](int class_id) {
            if (class_id == 0) return cv::Scalar(255, 0, 0);
            if (class_id == 1) return cv::Scalar(0, 255, 255);
            if (class_id == 2) return cv::Scalar(0, 165, 255);
            if (class_id == 3) return cv::Scalar(0, 0, 255);
            if (class_id == 4) return cv::Scalar(128, 128, 128);
            return cv::Scalar(255, 255, 255);
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

    if (export_stats_ && stats_file_.is_open()) {
        auto now = this->get_clock()->now();
        stats_file_ << now.nanoseconds() << "," << total_ms << "," << hz << "," << detections.size() << "\n";
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, "LATENCY: %.2f ms | FREQUENCY: %.2f Hz", total_ms, hz);
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
