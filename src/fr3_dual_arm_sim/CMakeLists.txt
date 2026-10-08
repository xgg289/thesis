cmake_minimum_required(VERSION 3.8)
project(fr3_dual_arm_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf worlds models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/sequential_dual_arm_6poses.py
  scripts/sequential_left_right_6poses.py
  scripts/moveit_node.py
  scripts/seq_dual_arm_move_group.py
  scripts/both_arms_gripper_twice.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()