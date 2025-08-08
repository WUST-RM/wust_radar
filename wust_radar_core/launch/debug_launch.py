import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory("wust_radar_core"), "config", "params.yaml"
    )

    return LaunchDescription(
        [
            Node(
                package="wust_radar_core",
                executable="wust_radar_core_node",
                name="wust_radar_core",
                parameters=[config_path],
                output="screen",
                prefix="gnome-terminal -- gdb -ex run --args",
            )
        ]
    )
