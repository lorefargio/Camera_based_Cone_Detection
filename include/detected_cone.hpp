#pragma once
#include <opencv2/opencv.hpp>

/**
 * @brief Structure representing a detected cone in 2D image space.
 * 
 * This structure holds the results of the YOLO segmentation, including
 * classification, confidence, and spatial location.
 */
struct DetectedCone {
    int class_id;              ///< YOLO Class ID (0: Blue, 1: Fallen, 2: Orange, 3: Big Orange, 4: Yellow)
    float yolo_confidence;     ///< Confidence score from the detector (0.0 - 1.0)
    cv::Point2f center_2d;     ///< 2D coordinates of the cone center in pixels
    cv::Rect bbox;             ///< Bounding box of the detection in pixels
    cv::Mat mask;              ///< Optional: binary mask for the specific instance
};
