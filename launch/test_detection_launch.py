import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('zed_fusion_perception')
    
    # Launch Configurations
    use_zed = LaunchConfiguration('use_zed')
    use_bag = LaunchConfiguration('use_bag')
    bag = LaunchConfiguration('bag')
    engine_path = LaunchConfiguration('engine_path')
    
    # 1. ZED Wrapper Launch
    # Viene lanciato solo se use_zed=true E use_bag=false
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('zed_wrapper'), 'launch', 'zed_camera.launch.py')
        ]),
        launch_arguments={
            'camera_model': 'zed2i'
        }.items(),
        condition=IfCondition(use_zed) and UnlessCondition(use_bag)
    )

    # 2. ROS Bag Play
    # Viene lanciato solo se use_bag=true
    # Esegue il comando: ros2 bag play <path_della_bag>
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag],
        output='screen',
        condition=IfCondition(use_bag)
    )

    # 3. Perception Node
    # Gira sempre, sia con ZED che con Bag
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
            # Se la bag ha topic diversi dai driver live, puoi aggiungere altri remapping qui
            ('/zed/zed_node/rgb/color/rect/image', '/zed/zed_node/rgb/color/rect/image'),
            ('/zed2i/zed_node/rgb/camera_info', '/zed2i/zed_node/rgb/camera_info'),
        ]
    )

    return LaunchDescription([
        # Argomenti del Launch
        DeclareLaunchArgument('use_zed', default_value='false', 
                              description='Lancia i driver fisici della ZED 2i'),
        
        DeclareLaunchArgument('use_bag', default_value='false', 
                              description='Lancia il replay di una rosbag invece della camera'),
        
        DeclareLaunchArgument('bag', default_value='', 
                              description='Percorso assoluto al file .mcap o alla cartella della rosbag'),
        
        DeclareLaunchArgument('engine_path', 
                              default_value=os.path.join(pkg_share, 'models', 'best_trained_on_real_data.engine'), 
                              description='Percorso al file TensorRT engine'),

        # Azioni
        zed_launch,
        bag_play,
        perception_node,
    ])