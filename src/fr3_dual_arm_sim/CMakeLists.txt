cmake_minimum_required(VERSION 3.8)
project(fr3_dual_arm_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf worlds models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/dual_arm_move_group_joint_pose1_pose2.py
  scripts/moveit_node.py
  scripts/seq_dual_arm_move_group.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()