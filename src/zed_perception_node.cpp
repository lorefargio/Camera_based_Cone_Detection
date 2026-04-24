#include "zed_fusion_perception/zed_perception_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <fstream>
#include <rclcpp_components/register_node_macro.hpp>

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
    RCLCPP_INFO(this->get_logger(), " - Mode: High-Performance Mask Provider");
    RCLCPP_INFO(this->get_logger(), " - Optimization: Single-Sync Batch Pipeline");
    RCLCPP_INFO(this->get_logger(), " - CUDA Graphs: Active");
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
        // ZERO-COPY: passing shared pointer instead of deep copy
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

    // 1. Launch recorded GPU pipeline (Graph: Preprocess -> Inference -> Reformat -> Mask)
    yolo_->infer_to_canvas(cv_ptr->image, d_mask_canvas);
    
    // 2. Launch asynchronous download of detection results for other logic
    auto detections = yolo_->infer(cv_ptr->image);

    // 3. Prepare ROS Image Message and launch asynchronous download of the canvas
    auto mask_msg = std::make_unique<sensor_msgs::msg::Image>();
    mask_msg->header = msg->header;
    mask_msg->height = cv_ptr->image.rows;
    mask_msg->width = cv_ptr->image.cols;
    mask_msg->encoding = "mono8";
    mask_msg->step = mask_msg->width;
    mask_msg->data.resize(canvas_size);
    
    // ASYNC DOWNLOAD: Trigger PCIe transfer without blocking CPU execution
    cudaMemcpyAsync(mask_msg->data.data(), d_mask_canvas, canvas_size, cudaMemcpyDeviceToHost, yolo_->getStream());

    // --- PIPELINE SYNCHRONIZATION PATH ---
    // Single barrier synchronization: overlapping computation and PCIe transfers
    cudaStreamSynchronize(yolo_->getStream());

    // From this point, CPU safely processes the data in RAM
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
            // Semantic Debug: Iterate over possible classes (1:Blue, 2:Yellow, etc.)
            for (int class_id = 0; class_id < 5; ++class_id) {
                color_mask.setTo(get_class_color(class_id), raw_mask == (class_id + 1));
            }
            debug_mask_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", color_mask).toImageMsg());
        }
    }

    mask_canvas_pub_->publish(std::move(mask_msg));

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
