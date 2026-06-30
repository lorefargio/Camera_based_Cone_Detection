#include "zed_perception_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <rclcpp_components/register_node_macro.hpp>

ZedPerceptionNode::ZedPerceptionNode(const rclcpp::NodeOptions& options)
    : Node("zed_perception_node", options) {
    
    // 1. Declare parameters
    this->declare_parameter("engine_path", "models/yolo26n-seg.engine");
    this->declare_parameter("conf_threshold", 0.5);
    this->declare_parameter("nms_threshold", 0.45);
    this->declare_parameter("publish_debug", false);
    this->declare_parameter("export_stats", false);
    this->declare_parameter("use_cuda_kernels", true);

    loadParametersToConfig();

    // 2. Instantiate pipeline and visualizer active object
    pipeline_ = std::make_unique<camera_perception::CameraPerceptionPipeline>();
    pipeline_->initialize(config_);

    viz_bridge_ = std::make_unique<camera_perception::CameraVisualizationBridge>(this);

    RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");
    RCLCPP_INFO(this->get_logger(), " CAMERA PERCEPTION NODE INITIALIZED (DECOUPLED)");
    RCLCPP_INFO(this->get_logger(), " - Mode: High-Performance Mask Provider");
    RCLCPP_INFO(this->get_logger(), " - Custom CUDA Kernels: %s", config_.use_cuda_kernels ? "ENABLED" : "DISABLED (CPU Baseline)");
    RCLCPP_INFO(this->get_logger(), " - Optimization: Active Object Visualizer Bridge");
    RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");

    if (config_.export_stats) {
        stats_file_.open("camera_stats.csv");
        stats_file_ << "timestamp,latency_ms,hz,detections,cuda_kernels_enabled\n";
    }

    // 3. Setup subscriptions
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/zed/zed_node/rgb/color/rect/image", 10, std::bind(&ZedPerceptionNode::imageCallback, this, std::placeholders::_1));

    info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/zed/zed_node/rgb/color/rect/camera_info", 10, std::bind(&ZedPerceptionNode::cameraInfoCallback, this, std::placeholders::_1));
}

ZedPerceptionNode::~ZedPerceptionNode() {
    if (viz_bridge_) {
        viz_bridge_->stop();
    }
    if (stats_file_.is_open()) {
        stats_file_.close();
    }
}

void ZedPerceptionNode::loadParametersToConfig() {
    config_.engine_path = this->get_parameter("engine_path").as_string();
    config_.conf_threshold = this->get_parameter("conf_threshold").as_double();
    config_.nms_threshold = this->get_parameter("nms_threshold").as_double();
    config_.publish_debug = this->get_parameter("publish_debug").as_bool();
    config_.export_stats = this->get_parameter("export_stats").as_bool();
    config_.use_cuda_kernels = this->get_parameter("use_cuda_kernels").as_bool();
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
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // 1. Run inference facade pipeline (downloads mask and coordinates asynchronously)
    std::vector<uint8_t> mask_data;
    auto detections = pipeline_->run(cv_ptr->image, config_, mask_data);

    // 2. Offload ROS message publishing and debug visuals to Active Object
    if (viz_bridge_) {
        viz_bridge_->enqueue(
            msg->header, 
            cv_ptr->image, 
            mask_data, 
            detections, 
            cv_ptr->image.rows, 
            cv_ptr->image.cols, 
            config_);
    }

    // 3. Profiling stats recording
    auto end_node = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_node - start_node).count();
    double hz = 1000.0 / total_ms;

    iter_count_++;
    if (config_.export_stats && stats_file_.is_open() && iter_count_ > 1) {
        auto now = this->get_clock()->now();
        stats_file_ << now.nanoseconds() << "," << total_ms << "," << hz << "," << detections.size() << "," << (config_.use_cuda_kernels ? 1 : 0) << "\n";
    }
    
    if (iter_count_ > 1) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, "LATENCY: %.2f ms | FREQUENCY: %.2f Hz", total_ms, hz);
    }
}


RCLCPP_COMPONENTS_REGISTER_NODE(ZedPerceptionNode)
