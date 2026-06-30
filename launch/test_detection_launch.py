import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('camera_perception')
    
    # Launch Configurations
    use_zed = LaunchConfiguration('use_zed')
    use_bag = LaunchConfiguration('use_bag')
    engine_path = LaunchConfiguration('engine_path')
    publish_debug = LaunchConfiguration('publish_debug')
    export_stats = LaunchConfiguration('export_stats')
    use_cuda_kernels = LaunchConfiguration('use_cuda_kernels')
    
    # 1. ZED Wrapper Launch
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('zed_wrapper'), 'launch', 'zed_camera.launch.py')
        ]),
        launch_arguments={
            'camera_model': 'zed2i'
        }.items(),
        condition=IfCondition(use_zed) and UnlessCondition(use_bag)
    )


    # 3. Perception Node
    perception_node = Node(
        package='camera_perception',
        executable='perception_node',
        name='zed_perception_node',
        output='screen',
        parameters=[{
            'engine_path': engine_path,
            'conf_threshold': 0.6,
            'nms_threshold': 0.45,
            'publish_debug': publish_debug,
            'export_stats': export_stats,
            'use_cuda_kernels': use_cuda_kernels,
        }],
        remappings=[
            ('/zed/zed_node/rgb/color/rect/image', '/zed/zed_node/rgb/color/rect/image'),
            ('/zed/zed_node/rgb/color/rect/camera_info', '/zed/zed_node/rgb/color/rect/camera_info'),
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_zed', default_value='false', 
                              description='Lancia i driver fisici della ZED 2i'),
        
        DeclareLaunchArgument('use_bag', default_value='true', 
                              description='Lancia il replay di una rosbag invece della camera'),
        
        DeclareLaunchArgument('engine_path', 
                              default_value=os.path.join(pkg_share, 'models', 'yolo26_fp16.engine'), 
                              description='Percorso al file TensorRT engine'),
        
        DeclareLaunchArgument('publish_debug', default_value='true', 
                              description='Pubblica immagini di debug per visualizzazione'),
        
        DeclareLaunchArgument('export_stats', default_value='false', 
                              description='Esporta i tempi di esecuzione in perception_stats.csv'),
        
        DeclareLaunchArgument('use_cuda_kernels', default_value='true', 
                              description='Abilita l\'uso dei custom CUDA kernels (src/cuda_kernels.cu) per preprocessing e postprocessing'),

        zed_launch,
        perception_node,
    ])
