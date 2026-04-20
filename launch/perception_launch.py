import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='zed_fusion_perception',
            executable='zed_fusion_node',
            name='zed_fusion_perception_node',
            output='screen',
            parameters=[{
                'engine_path': 'models/yolo26n-seg.engine',
                'conf_threshold': 0.5,
                'nms_threshold': 0.45,
                'spatial_threshold': 0.6
            }],
            remappings=[
                ('/zed2i/zed_node/rgb/image_rect_color', '/zed2i/zed_node/rgb/image_rect_color'),
                ('/zed2i/zed_node/rgb/camera_info', '/zed2i/zed_node/rgb/camera_info'),
                ('/lidar_perception/clusters', '/lidar_perception/clusters')
            ]
        )
    ])
