from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    mpc_node = Node(
        package="mpc_controller",
        executable="mpc_node",
        name="mpc_node",
        output="screen",
    )

    return LaunchDescription([mpc_node])
