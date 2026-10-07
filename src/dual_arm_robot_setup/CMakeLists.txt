cmake_minimum_required(VERSION 3.8)
project(dual_arm_robot_setup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf worlds models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()