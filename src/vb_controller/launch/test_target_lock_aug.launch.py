from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    declare_scenario = DeclareLaunchArgument(
        "scenario",
        default_value="1",
        description="Test scenario: 1=linear_forward, 2=driveby, 3=circular, 4=sinusoidal",
    )
    declare_duration = DeclareLaunchArgument(
        "duration",
        default_value="10.0",
        description="Duration of the test run in seconds",
    )

    declare_model = DeclareLaunchArgument(
        "model",
        default_value="NCA",
        description="Tracker model under test: NCA (base) or NCA_AUG (augmented)",
    )

    test_target_lock_node = Node(
        package="vb_controller",
        executable="test_target_lock_aug.py",
        name="test_target_lock_node",
        output="screen",
        parameters=[
            {
                "scenario": ParameterValue(
                    LaunchConfiguration("scenario"), value_type=int
                ),
                "duration": ParameterValue(
                    LaunchConfiguration("duration"), value_type=float
                ),
                "model": ParameterValue(LaunchConfiguration("model"), value_type=str),
            }
        ],
    )

    return LaunchDescription(
        [declare_scenario, declare_duration, declare_model, test_target_lock_node]
    )
