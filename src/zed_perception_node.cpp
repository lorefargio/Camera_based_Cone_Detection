#include "zed_perception_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <fstream>
#include <rclcpp_components/register_node_macro.hpp>

ZedPerceptionNode::ZedPerceptionNode(const rclcpp::NodeOptions& options)
    : Node("zed_perception_node", options) {
    
    // Declare and retrieve parameters
    std::string engine_path = this->declare_parameter("engine_path", "models/yolo26n-seg.engine");
    float conf_threshold = this->declare_parameter("conf_threshold", 0.5);
    float nms_threshold = this->declare_parameter("nms_threshold", 0.45);
    publish_debug_ = this->declare_parameter("publish_debug", false);
    export_stats_ = this->declare_parameter("export_stats", false);
    bool use_cuda_kernels = this->declare_parameter("use_cuda_kernels", true);

    // Initialize inference engine
    yolo_ = std::make_unique<Yolo26nSeg>(engine_path, conf_threshold, nms_threshold, use_cuda_kernels);

    RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");
    RCLCPP_INFO(this->get_logger(), " CAMERA PERCEPTION NODE INITIALIZED");
    RCLCPP_INFO(this->get_logger(), " - Mode: High-Performance Mask Provider");
    RCLCPP_INFO(this->get_logger(), " - Custom CUDA Kernels: %s", use_cuda_kernels ? "ENABLED" : "DISABLED (CPU Baseline)");
    RCLCPP_INFO(this->get_logger(), " - Optimization: Single-Sync Batch Pipeline");
    RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");

    if (export_stats_) {
        stats_file_.open("camera_stats.csv");
        stats_file_ << "timestamp,latency_ms,hz,detections\n";
    }

    // Initialize subscriptions
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/rgb/color/rect/image", 10, std::bind(&ZedPerceptionNode::imageCallback, this, std::placeholders::_1));

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/zed/zed_node/rgb/color/rect/camera_info", 10, std::bind(&ZedPerceptionNode::cameraInfoCallback, this, std::placeholders::_1));

    // Initialize publishers
    debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/debug_image", 10);
    debug_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/debug_mask_canvas", 10);
    mask_canvas_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/perception/camera_mask_canvas", 10);
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
        // Use toCvShare for zero-copy if the encoding matches
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Manage persistent GPU mask buffer
    static uint8_t* d_mask_canvas = nullptr;
    static size_t canvas_size = 0;
    if (d_mask_canvas == nullptr || canvas_size != cv_ptr->image.total()) {
        if (d_mask_canvas) cudaFree(d_mask_canvas);
        canvas_size = cv_ptr->image.total();
        cudaMalloc(reinterpret_cast<void**>(&d_mask_canvas), canvas_size);
    }

    // 1. Execute full GPU pipeline
    yolo_->infer_to_canvas(cv_ptr->image, d_mask_canvas);
    
    // 2. Trigger asynchronous download of metadata (detections)
    auto detections = yolo_->infer(cv_ptr->image);

    // 3. Prepare output ROS Image Message
    auto mask_msg = std::make_unique<sensor_msgs::msg::Image>();
    mask_msg->header = msg->header; // Precise timestamp preservation
    mask_msg->height = cv_ptr->image.rows;
    mask_msg->width = cv_ptr->image.cols;
    mask_msg->encoding = "mono8";
    mask_msg->step = mask_msg->width;
    mask_msg->data.resize(canvas_size);
    
    // Trigger asynchronous download of the mask canvas
    cudaMemcpyAsync(mask_msg->data.data(), d_mask_canvas, canvas_size, cudaMemcpyDeviceToHost, yolo_->getStream());

    // --- SINGLE BARRIER SYNCHRONIZATION ---
    cudaStreamSynchronize(yolo_->getStream());

    // Debug visualization path
    if (publish_debug_) {
        cv::Mat raw_mask(mask_msg->height, mask_msg->width, CV_8U, (void*)mask_msg->data.data());
        auto get_class_color = [](int class_id) {
            if (class_id == 0) return cv::Scalar(255, 0, 0);   // Blue
            if (class_id == 1) return cv::Scalar(0, 255, 255); // Yellow
            if (class_id == 2) return cv::Scalar(0, 165, 255); // Orange
            if (class_id == 3) return cv::Scalar(0, 0, 255);   // Big Orange (Red)
            if (class_id == 4) return cv::Scalar(128, 128, 128); // Fallen (Gray)
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
            for (int class_id = 0; class_id < 5; ++class_id) {
                color_mask.setTo(get_class_color(class_id), raw_mask == (class_id + 1));
            }
            debug_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", color_mask).toImageMsg());
        }
    }

    // Publish the mask message (using std::move for Zero-Copy if supported by transport)
    mask_canvas_pub_->publish(std::move(mask_msg));

    // Performance metrics calculation
    auto end_node = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_node - start_node).count();
    double hz = 1000.0 / total_ms;

    static int iter_count = 0;
    iter_count++;
    if (export_stats_ && stats_file_.is_open() && iter_count > 1) {
        auto now = this->get_clock()->now();
        stats_file_ << now.nanoseconds() << "," << total_ms << "," << hz << "," << detections.size() << "\n";
    }
    
    if (iter_count > 1) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, "LATENCY: %.2f ms | FREQUENCY: %.2f Hz", total_ms, hz);
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(ZedPerceptionNode)
