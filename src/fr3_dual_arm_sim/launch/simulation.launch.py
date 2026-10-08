import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    RegisterEventHandler,
    GroupAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node, SetParameter

import xacro

def generate_launch_description():

    # ------------------------
    # Simulation package
    # ------------------------

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
    world_path_arg = DeclareLaunchArgument(
        'world_path',
        default_value=os.path.join(
            pkg_dual_arm,
            'worlds',
            'simulation_world.sdf'
        ),
        description='Path to the Gazebo world file'
    )
    # -----------------------------
    # Dual FR3 Xacro
    # -----------------------------
    xacro_file = os.path.join(
        pkg_moveit_description,
        'config',
        'dual_fr3.urdf.xacro'
    )
    robot_description_config = xacro.process_file(
        xacro_file
    )

    robot_description = {
        'robot_description':
        robot_description_config.toxml()
    }

    srdf_file = os.path.join(
        pkg_moveit_description,
        'config',
        'dual_fr3.srdf'
    )

    with open(srdf_file, 'r') as f:
        robot_description_semantic = {
            'robot_description_semantic': f.read()
        }

    # -----------------------------
    #Gazebo
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
            'gz_args': [
                LaunchConfiguration('world_path'),
                ' -r -v 4',
                ' --physics-engine gz-physics-bullet-featherstone-plugin'
            ]
        }.items()
    )

    # -----------------------------
    # Gazebo -> ROS clock
    # -----------------------------
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen'
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
    left_hand_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_fr3_hand_controller'],
        output='screen'
    )

    right_hand_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_fr3_hand_controller'],
        output='screen'
    )

    # -----------------------------
    # MoveIt
    # -----------------------------
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_moveit_description,
                'launch',
                'move_group.launch.py'
            )
        )
    )

    # -----------------------------
    # RViz
    # -----------------------------
    moveit_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_moveit_description,
                'launch',
                'moveit_rviz.launch.py'
            )
        )
    )

    # -----------------------------
    # Gazebo simulation time
    # -----------------------------
    moveit_and_rviz = GroupAction([
        SetParameter(
            name='use_sim_time',
            value=True
        ),

        move_group,
        moveit_rviz
    ])

    # ------------------------------------------------
    # Startup order:
    #
    # robot spawned
    #      ↓
    # joint_state_broadcaster
    #      ↓
    # left arm controller
    #      ↓
    # right arm controller
    #      ↓
    # MoveIt + RViz
    # ------------------------------------------------

    start_joint_state_broadcaster = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                joint_state_broadcaster_spawner
            ]
        )
    )

    start_all_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[
                left_arm_controller_spawner,
                right_arm_controller_spawner,
                left_hand_controller_spawner,
                right_hand_controller_spawner,
            ]
        )
    )

    # -----------------------------
    # Launch
    # -----------------------------
    return LaunchDescription([

        # Allow Gazebo to find Franka meshes/resources
        AppendEnvironmentVariable(
            name='GZ_SIM_RESOURCE_PATH',
            value=os.path.dirname(pkg_franka_description)
        ),

        world_path_arg,
        
        gazebo,

        gz_bridge,

        robot_state_publisher,

        spawn_robot,

        start_joint_state_broadcaster,

        start_all_controllers,

        moveit_and_rviz
        
    ])