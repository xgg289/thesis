cmake_minimum_required(VERSION 3.8)
project(fr3_dual_arm_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf worlds models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/rq1/seq_exp1.py
  scripts/rq1/seq_exp2.py
  scripts/rq1/seq_exp3.py
  scripts/rq1/seq_exp4.py
  scripts/trial.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()