import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ConditionalAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('zed_fusion_perception')
    
    # Launch Arguments
    use_zed = LaunchConfiguration('use_zed', default='false')
    use_foxglove = LaunchConfiguration('use_foxglove', default='true')
    engine_path = LaunchConfiguration('engine_path', default=os.path.join(pkg_share, 'models', 'yolo26n-seg.engine'))
    
    # ZED Wrapper Launch (requires zed_wrapper to be installed)
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('zed_wrapper'), 'launch', 'zed2i.launch.py')
        ]),
        condition=PythonExpression([use_zed, ' == "true"'])
    )

    # Perception Node
    perception_node = Node(
        package='zed_fusion_perception',
        executable='zed_fusion_node',
        name='zed_perception_node',
        output='screen',
        parameters=[{
            'engine_path': engine_path,
            'conf_threshold': 0.5,
            'nms_threshold': 0.45,
        }],
        remappings=[
            # Remap if bag topics are different, but these are standard ZED topics
            ('/zed2i/zed_node/rgb/image_rect_color', '/zed2i/zed_node/rgb/image_rect_color'),
            ('/zed2i/zed_node/rgb/camera_info', '/zed2i/zed_node/rgb/camera_info'),
        ]
    )

    # Foxglove Bridge Node
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge_node',
        name='foxglove_bridge',
        condition=PythonExpression([use_foxglove, ' == "true"'])
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_zed', default_value='false', description='Launch ZED camera wrapper'),
        DeclareLaunchArgument('use_foxglove', default_value='true', description='Launch Foxglove bridge'),
        DeclareLaunchArgument('engine_path', default_value=os.path.join(pkg_share, 'models', 'yolo26n-seg.engine'), description='Path to TensorRT engine'),
        
        zed_launch,
        perception_node,
        foxglove_bridge
    ])
