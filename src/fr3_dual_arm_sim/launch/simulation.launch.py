import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

import xacro


def generate_launch_description():

    # Simulation package
    pkg_dual_arm = get_package_share_directory(
        'fr3_dual_arm_sim'
    )

    # MoveIt / ros2_control package
    pkg_moveit_description = get_package_share_directory(
        'fr3_dual_arm_moveit_config'
    )

    # Gazebo ROS package
    pkg_ros_gz_sim = get_package_share_directory(
        'ros_gz_sim'
    )

    # Franka description package
    pkg_franka_description = get_package_share_directory(
        'franka_description'
    )

    # -----------------------------
    # World
    # -----------------------------
    world_file = os.path.join(
        pkg_dual_arm,
        'worlds',
        'simulation_world.sdf'
    )

    # -----------------------------
    # Dual FR3 Xacro
    # -----------------------------
    xacro_file = os.path.join(
        pkg_moveit_description,
        'config',
        'dual_fr3.urdf.xacro'
    )

    robot_description_xml = xacro.process_file(
        xacro_file
    ).toxml()

    robot_description = {
        'robot_description': robot_description_xml
    }

    # -----------------------------
    # Start Gazebo
    # -----------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_ros_gz_sim,
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': f'{world_file} -r'
        }.items()
    )

    # -----------------------------
    # Robot State Publisher
    # -----------------------------
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description,
            {'use_sim_time': True}
        ]
    )

    # -----------------------------
    # Spawn dual FR3 in Gazebo
    # -----------------------------
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic',
            'robot_description',
            '-name',
            'dual_fr3'
        ],
        output='screen'
    )
    # -----------------------------
    # Controller Spawners
    # -----------------------------
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    left_arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_fr3_arm_controller'],
        output='screen'
    )

    right_arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_fr3_arm_controller'],
        output='screen'
    )
    return LaunchDescription([

        # Allow Gazebo to find Franka meshes/resources
        AppendEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=os.path.dirname(pkg_franka_description)
        ),

        gazebo,

        robot_state_publisher,

        spawn_robot,

        joint_state_broadcaster_spawner,

        left_arm_controller_spawner,

        right_arm_controller_spawner
    ])