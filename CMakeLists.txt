cmake_minimum_required(VERSION 3.8)
project(zed_fusion_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Find System Dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET)

# TensorRT must be found manually or via a FindTensorRT.cmake module
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_LIBRARY nvinfer)
find_library(TENSORRT_PARSER nvonnxparser)

if (CUDA_FOUND AND TENSORRT_LIBRARY)
    message(STATUS "CUDA and TensorRT found. Enabling GPU features.")
    include_directories(include ${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

    add_library(zed_fusion_core SHARED
      src/yolo26_tensorrt.cpp
      src/zed_perception_node.cpp
    )
    target_link_libraries(zed_fusion_core
      ${OpenCV_LIBRARIES}
      ${TENSORRT_LIBRARY}
      ${TENSORRT_PARSER}
      ${CUDA_CUDART_LIBRARY}
    )
    ament_target_dependencies(zed_fusion_core
      rclcpp sensor_msgs geometry_msgs visualization_msgs cv_bridge image_transport tf2_ros tf2_geometry_msgs
    )

    add_executable(zed_fusion_node src/main.cpp)
    target_link_libraries(zed_fusion_node zed_fusion_core)

    add_executable(test_yolo_standalone src/test_yolo_standalone.cpp)
    target_link_libraries(test_yolo_standalone zed_fusion_core)

    install(TARGETS zed_fusion_core zed_fusion_node test_yolo_standalone
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION lib/${PROJECT_NAME}
    )
else()
    message(WARNING "CUDA or TensorRT NOT found. Compilation will likely fail if attempted.")
endif()
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)

ament_package()
