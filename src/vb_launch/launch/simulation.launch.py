#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Generate launch description for IKA simulation setup.

    This launch file starts:
    - Core system in simulation mode
    - Gazebo simulation with IKA robot
    - ROS2 control stack
    - Vision processing nodes
    - Action servers
    - Navigation stack
    - Mission planning
    - User interface
    """

    # Core node - Launch in SIMULATION mode, MANUAL state, without mobility
    # Gazebo and Rviz launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("vb_description"),
                        "launch",
                        "ika_gazebo.launch.py",
                    ]
                )
            ]
        ),
        launch_arguments={"world_name": "shoot_parkour"}.items(),
    )

    # ROS2 Control launch
    controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("vb_controller"),
                        "launch",
                        "vb_controllers.launch.py",
                    ]
                )
            ]
        )
    )

    lock_target_node = Node(
        package="vb_actions", executable="lock_target.py", name="lock_target"
    )

    return LaunchDescription(
        [
            gazebo_launch,
            controllers_launch,
            # lock_target_node,
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
