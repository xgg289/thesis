# Dual-arm FR3 simulation

This workspace contains a Gazebo simulation of two Franka FR3 manipulators controlled through ROS 2 Jazzy and MoveIt 2. The experiments compare independent and combined planning, sequential and simultaneous execution, and independent planning with combined fallback. Scenarios include separate regions, a shared workspace, a single divider, four obstacles, and a practical tray-and-block manipulation task.

## Simulation environment

The following versions were verified in the current simulation environment, with the `franka_ros2` workspace sourced. Package versions for earlier recorded trials were not separately captured.

| Component | Version |
|---|---|
| Ubuntu | 24.04.5 LTS (Noble Numbat) |
| ROS 2 | Jazzy Jalisco |
| ROS desktop metapackage (`ros-jazzy-desktop`) | `0.11.0-1noble.20261006.040109` |
| Gazebo Harmonic (`gz-sim8`) | 8.15.0 |
| Gazebo Physics (`gz-physics7`, Bullet Featherstone backend) | 7.8.0 |
| ROS–Gazebo simulation and clock bridge (`ros_gz_sim`, `ros_gz_bridge`) | 1.0.24 |
| MoveIt 2 | 2.12.4 |
| OMPL | 1.7.0 |
| RViz 2 | 14.1.24 |
| `ros2_control` / controller manager | 4.49.0 |
| `ros2_controllers` / arm and gripper controllers | 4.43.0 |
| `gz_ros2_control` | 1.2.20 |
| `franka_ros2` source checkout | `v3.1.1`, commit `cc6d691eb3160811e553bde6f9820987b2ffe6f2`, with local modifications |
| `franka_ros2` / `franka_gz_ros2_control` package manifests | 3.1.0 |
| `franka_description` | 2.9.0 |
| `libfranka` | 0.20.5 |
| Robot State Publisher | 3.3.4 |
| Xacro | 2.1.1 |
| ROS Python client (`rclpy`) | 7.1.12 |
| Python | 3.12.3 |

## Setup

The workspace requires ROS 2 Jazzy, Gazebo Harmonic, MoveIt 2, and a built `franka_ros2` workspace. The commands below assume the workspaces are located at `~/dual_arm` and `~/franka_ros2_ws`.

Build the workspace:

```bash
cd ~/dual_arm
source /opt/ros/jazzy/setup.bash
source ~/franka_ros2_ws/install/local_setup.bash
rosdep install --from-paths src --ignore-src --rosdistro jazzy -y
colcon build
source install/local_setup.bash
```

Source these environments in **each new terminal** before launching the simulation or running an experiment:

```bash
cd ~/dual_arm
source /opt/ros/jazzy/setup.bash
source ~/franka_ros2_ws/install/local_setup.bash
source install/local_setup.bash
```

## Launch the simulation

In the first terminal, select the world using `world_path`. There is no need to edit `simulation.launch.py` when switching scenarios.

For experiments without obstacles:

```bash
ros2 launch fr3_dual_arm_sim simulation.launch.py \
  world_path:="$PWD/src/fr3_dual_arm_sim/worlds/simulation_world.sdf"
```

For a single divider, select the file matching the experiment set:

```bash
# Divider set 1; replace _1 with _2 or _3 for the other sets.
ros2 launch fr3_dual_arm_sim simulation.launch.py \
  world_path:="$PWD/src/fr3_dual_arm_sim/worlds/simulation_world_obstacle1_1.sdf"
```

For four obstacles:

```bash
ros2 launch fr3_dual_arm_sim simulation.launch.py \
  world_path:="$PWD/src/fr3_dual_arm_sim/worlds/simulation_world_obstacle2.sdf"
```

For the practical demonstration:

```bash
ros2 launch fr3_dual_arm_sim simulation.launch.py \
  world_path:="$PWD/src/fr3_dual_arm_sim/worlds/practical_world.sdf"
```

The launch file starts Gazebo, the robot description, the simulation-clock bridge, the arm and gripper controllers, MoveIt, and RViz. Experiment movements start when their Python script is run separately.

Stop the current simulation with **Ctrl+C** before launching a different world. Start a fresh simulation for each recorded trial and each practical demonstration.

## Run an experiment

Open a second terminal and source the environments shown above. Wait for the robot, MoveIt, and controllers to finish initializing. Run **one experiment command at a time** in its matching world.

### Research Question 1: planning and execution comparison

Use the obstacle-free world. The following commands select set 1:

```bash
# Sequential execution: independent or combined planning.
ros2 run fr3_dual_arm_sim seq_ind_1.py
ros2 run fr3_dual_arm_sim seq_com_1.py

# Simultaneous execution: independent or combined planning.
ros2 run fr3_dual_arm_sim sim_ind_1.py
ros2 run fr3_dual_arm_sim sim_com_1.py
```

Replace the final `_1.py` with `_2.py` or `_3.py` to select another configuration set.

### Research Question 2: workspace and obstacle comparison

The commands below select set 1. Replace the final `_1.py` with `_2.py` or `_3.py` for the other sets.

```bash
# Separate regions: obstacle-free world.
ros2 run fr3_dual_arm_sim ind_sep_reg_1.py
ros2 run fr3_dual_arm_sim com_sep_reg_1.py

# Shared workspace: obstacle-free world.
ros2 run fr3_dual_arm_sim ind_sdr_ws_1.py
ros2 run fr3_dual_arm_sim com_sdr_ws_1.py

# Shared workspace with four obstacles: simulation_world_obstacle2.sdf.
ros2 run fr3_dual_arm_sim ind_sdr_ws_4_obs_1.py
ros2 run fr3_dual_arm_sim com_sdr_ws_4_obs_1.py

# Shared workspace with one divider: matching divider world for the selected set.
ros2 run fr3_dual_arm_sim ind_sdr_ws_div_obs_1.py
ros2 run fr3_dual_arm_sim com_sdr_ws_div_obs_1.py
```

### Research Question 3: independent planning with combined fallback

Independent planning is attempted first. Combined fallback is attempted if the independent candidate fails planning, preparation, or validation; fallback activation and successful completion are not guaranteed.

```bash
# Four-obstacle world: 
ros2 run fr3_dual_arm_sim 4_obs1_fallback.py
ros2 run fr3_dual_arm_sim 4_obs2_fallback.py
ros2 run fr3_dual_arm_sim 4_obs3_fallback.py

# Single divider world: 
ros2 run fr3_dual_arm_sim 1_divider1_fallback.py
ros2 run fr3_dual_arm_sim 1_divider2_fallback.py
ros2 run fr3_dual_arm_sim 1_divider3_fallback.py
```

### Practical demonstration

Launch a fresh `practical_world.sdf`, then run:

```bash
ros2 run fr3_dual_arm_sim practical.py
```

The left arm picks up a tray, and the right arm places a block inside it. The arms then withdraw and return the loaded tray to its original pickup stand. The demonstration reuses the fallback planner from `rq3/4_obs1_fallback.py`.

## Results 
Scripts display timing measurements, task outcomes, and available collision details in the terminal. In fallback experiments, a rejected independent plan can still lead to task success if combined planning succeeds.
