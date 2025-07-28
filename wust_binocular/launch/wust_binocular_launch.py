import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory("wust_binocular"), "config", "params.yaml"
    )

    return LaunchDescription(
        [
            Node(
                package="wust_binocular",
                executable="wust_binocular_node",
                name="wust_binocular",
                parameters=[config_path],
                output="screen",
                # prefix="gnome-terminal -- gdb -ex run --args",
            )
        ]
    )
