#include <iostream>
#include <opencv2/opencv.hpp>
#include "zed_fusion_perception/yolo26_tensorrt.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
        return -1;
    }

    std::string engine_path = argv[1];
    std::string image_path = argv[2];

    try {
        Yolo26nSeg yolo(engine_path);
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "Could not read image: " << image_path << std::endl;
            return -1;
        }

        auto detections = yolo.infer(img);
        std::cout << "Detected " << detections.size() << " cones." << std::endl;

        for (const auto& det : detections) {
            std::cout << "- Class: " << det.class_id 
                      << " Conf: " << det.yolo_confidence 
                      << " Area: " << det.mask_area << std::endl;
            
            // Visualization
            cv::circle(img, det.center_2d, 5, cv::Scalar(0, 255, 0), -1);
            if (!det.mask.empty()) {
                cv::Mat mask_color;
                cv::cvtColor(det.mask, mask_color, cv::COLOR_GRAY2BGR);
                cv::addWeighted(img, 1.0, mask_color, 0.4, 0, img);
            }
        }

        cv::imwrite("test_output.jpg", img);
        std::cout << "Output saved to test_output.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
