cmake_minimum_required(VERSION 3.8)
project(fr3_dual_arm_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf worlds models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/rq1/seq_com_1.py
  scripts/rq1/seq_com_2.py
  scripts/rq1/seq_com_3.py
  scripts/rq1/seq_ind_1.py
  scripts/rq1/seq_ind_2.py
  scripts/rq1/seq_ind_3.py
  scripts/rq1/sim_com_1.py
  scripts/rq1/sim_com_2.py
  scripts/rq1/sim_com_3.py
  scripts/rq1/sim_ind_1.py
  scripts/rq1/sim_ind_2.py
  scripts/rq1/sim_ind_3.py
  scripts/rq2/set_1/com_sdr_ws_1.py
  scripts/rq2/set_1/com_sdr_ws_4_obs_1.py
  scripts/rq2/set_1/com_sdr_ws_div_obs_1.py
  scripts/rq2/set_1/com_sep_reg_1.py
  scripts/rq2/set_1/ind_sdr_ws_1.py
  scripts/rq2/set_1/ind_sdr_ws_4_obs_1.py
  scripts/rq2/set_1/ind_sdr_ws_div_obs_1.py
  scripts/rq2/set_1/ind_sep_reg_1.py
  scripts/rq2/set_2/com_sdr_ws_2.py
  scripts/rq2/set_2/com_sdr_ws_4_obs_2.py
  scripts/rq2/set_2/com_sdr_ws_div_obs_2.py
  scripts/rq2/set_2/com_sep_reg_2.py
  scripts/rq2/set_2/ind_sdr_ws_2.py
  scripts/rq2/set_2/ind_sdr_ws_4_obs_2.py
  scripts/rq2/set_2/ind_sdr_ws_div_obs_2.py
  scripts/rq2/set_2/ind_sep_reg_2.py
  scripts/rq2/set_3/com_sdr_ws_3.py
  scripts/rq2/set_3/com_sdr_ws_4_obs_3.py
  scripts/rq2/set_3/com_sdr_ws_div_obs_3.py
  scripts/rq2/set_3/com_sep_reg_3.py
  scripts/rq2/set_3/ind_sdr_ws_3.py
  scripts/rq2/set_3/ind_sdr_ws_4_obs_3.py
  scripts/rq2/set_3/ind_sdr_ws_div_obs_3.py
  scripts/rq2/set_3/ind_sep_reg_3.py
  scripts/rq3/4_obs1_fallback.py
  scripts/rq3/4_obs2_fallback.py
  scripts/rq3/4_obs3_fallback.py
  scripts/rq3/1_divider1_fallback.py
  scripts/rq3/1_divider2_fallback.py
  scripts/rq3/1_divider3_fallback.py
  scripts/practical.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()