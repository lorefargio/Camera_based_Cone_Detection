#include "yolo26_tensorrt.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

/**
 * @brief Standalone utility to test the perception pipeline on a video file.
 * 
 * This tool allows for verification of the CUDA pipeline and TensorRT engine
 * without requiring a running ROS 2 environment or ZED camera.
 */
int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <path_engine> <path_video_input>"
              << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string video_path = argv[2];

  try {
    // 1. Initialize the YOLO model
    Yolo26nSeg yolo(engine_path, 0.45, 0.45); 

    // 2. Open input video stream
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::cerr << "Error: Could not open video file: " << video_path
                << std::endl;
      return -1;
    }

    // 3. Setup VideoWriter for saving results
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("output_result.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                           cv::Size(width, height));

    std::cout << "Starting video processing (" << width << "x" << height
              << " @ " << fps << "fps)..." << std::endl;

    cv::Mat frame;
    int frame_count = 0;

    while (cap.read(frame)) {
      // --- INFERENCE ---
      auto start = std::chrono::high_resolution_clock::now();
      auto detections = yolo.infer(frame);
      auto end = std::chrono::high_resolution_clock::now();

      // --- VISUALIZATION ---
      for (const auto &det : detections) {
        // Assign color based on class
        cv::Scalar color;
        if (det.class_id == 0)
          color = cv::Scalar(255, 0, 0);   // Blue
        else if (det.class_id == 1)
          color = cv::Scalar(128, 128, 128); // Fallen (Gray)
        else if (det.class_id == 2)
          color = cv::Scalar(0, 165, 255); // Orange
        else if (det.class_id == 3)
          color = cv::Scalar(0, 0, 255);   // Big Orange (Red)
        else if (det.class_id == 4)
          color = cv::Scalar(0, 255, 255); // Yellow
        else
          color = cv::Scalar(0, 255, 0);   // Unknown

        // Draw detection center
        cv::circle(frame, det.center_2d, 4, color, -1);
      }

      // Write processed frame to output
      writer.write(frame);

      // Log progress periodically
      double ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      if (frame_count % 30 == 0) {
        std::cout << "Frame: " << frame_count << " | Inference Time: " << ms
                  << "ms" << std::endl;
      }
      frame_count++;
    }

    std::cout
        << "Processing complete! Result saved to: output_result.mp4"
        << std::endl;
    cap.release();
    writer.release();

  } catch (const std::exception &e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
