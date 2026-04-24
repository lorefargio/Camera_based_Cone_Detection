cmake_minimum_required(VERSION 3.8)
project(zed_fusion_perception)

# Enable CUDA
enable_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Find System Dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT must be found manually
find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
find_library(TENSORRT_LIBRARY nvinfer)
find_library(TENSORRT_PARSER nvonnxparser)

if (CUDA_FOUND AND TENSORRT_LIBRARY)
    message(STATUS "CUDA and TensorRT found. Enabling GPU features.")
    include_directories(include ${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

    add_library(zed_fusion_core SHARED
      src/yolo26_tensorrt.cpp
      src/zed_perception_node.cpp
      src/cuda_kernels.cu
    )
    
    set_target_properties(zed_fusion_core PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES "75" # T1000 is Turing (7.5)
    )

    target_link_libraries(zed_fusion_core
      ${OpenCV_LIBRARIES}
      ${TENSORRT_LIBRARY}
      ${TENSORRT_PARSER}
      ${CUDA_CUDART_LIBRARY}
      cuda
    )
    ament_target_dependencies(zed_fusion_core
      rclcpp rclcpp_components sensor_msgs geometry_msgs visualization_msgs cv_bridge image_transport tf2_ros tf2_geometry_msgs
    )

    rclcpp_components_register_node(zed_fusion_core
      PLUGIN "ZedPerceptionNode"
    )

    add_executable(zed_fusion_node src/main.cpp)
    target_link_libraries(zed_fusion_node zed_fusion_core)

    add_executable(video_tester src/test_video.cpp src/yolo26_tensorrt.cpp src/cuda_kernels.cu)
    set_target_properties(video_tester PROPERTIES CUDA_ARCHITECTURES "75")
    target_link_libraries(video_tester
      ${OpenCV_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${TENSORRT_LIBRARY}
      cuda
    )
    ament_target_dependencies(video_tester
      rclcpp geometry_msgs sensor_msgs cv_bridge
    )

    install(TARGETS zed_fusion_core zed_fusion_node video_tester
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION lib/${PROJECT_NAME}
    )
else()
    message(WARNING "CUDA or TensorRT NOT found. Compilation will likely fail.")
endif()

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)

ament_package()