#pragma once
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/point.hpp>

struct DetectedCone {
    int class_id; // e.g., 0: blue, 1: yellow, 2: orange
    float yolo_confidence;
    cv::Point2f center_2d; // Center of the bounding box
    double mask_area; // Area (pixels) of the segmented cone
    cv::Mat mask; // Binary mask
    geometry_msgs::msg::Point position_3d; // Estimated 3D position
    float final_confidence; // 1.0 (LiDAR), 0.8 (Fused), 0.2 (Camera only)
};
