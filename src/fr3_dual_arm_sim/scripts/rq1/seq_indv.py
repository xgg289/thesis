#!/usr/bin/env python3
#Research Question 1: Independent planning with sequential execution.

import bisect
import copy
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from action_msgs.msg import GoalStatus
from moveit_msgs.srv import (GetMotionPlan, GetStateValidity, GetPlanningScene,
                             ApplyPlanningScene, SetPlannerParams, GetPlannerParams)
from moveit_msgs.msg import (MotionPlanRequest, Constraints, JointConstraint, RobotState,
                             PlanningSceneComponents)


class DualArmMoveGroupJointPose(Node):
    """One arm moves while the other holds its position."""

    def __init__(self):
        super().__init__("independent_sequential")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        # Trial uses simulation_world.sdf.
        self.latest_joint_state = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        # Services handle planning and collision checks.
        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.validity_client = self.create_client(GetStateValidity, "/check_state_validity")
        self.scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self.apply_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.exec_clients = [ActionClient(self, FollowJointTrajectory,
            f"/{side}_fr3_arm_controller/follow_joint_trajectory")
            for side in ("left", "right")]
        self.check_dt = 0.02
        self.check_joint_step = 0.01
        self.start_delay = 0.5  # Both controllers get time to accept goals.
        self.velocity_scale = 1.0
        self.acceleration_scale = 1.0
        self.joint_limits = {}
        self.limit_client = self.create_client(GetParameters, "/move_group/get_parameters")

        base = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]

        def pose(arm, joint1, joint3):
            joints = base.copy()
            joints[0] = joint1
            joints[2] = joint3
            return {
                f"{arm}_fr3_joint{i}": value
                for i, value in enumerate(joints, start=1)
            }

        # HOME positions.
        self.left_home = pose("left", 0.0, 0.0)
        self.right_home = pose("right", 0.0, 0.0)

        # Left arm targets.
        self.left_pose1 = pose("left", -0.8, 0.3)
        self.left_pose2 = pose("left",  0.0, -0.4)
        self.left_pose3 = pose("left",  0.8, 0.3)

        # Right arm targets.
        self.right_pose4 = pose("right",  0.8, -0.3)
        self.right_pose5 = pose("right",  0.0, 0.4)
        self.right_pose6 = pose("right", -0.8, -0.3)

    def _js_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_joint_state(self, timeout_sec=15.0) -> bool:
        # Arms waiting 0.3 seconds before planning.
        self.latest_joint_state = None
        required = set(self.left_home) | set(self.right_home)
        t0 = time.monotonic()
        stable_since, reference, previous = None, None, None
        while rclpy.ok() and time.monotonic() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            js = self.latest_joint_state
            if js is None or js is previous:
                continue
            previous = js
            if (len(js.name) != len(js.position) or len(set(js.name)) != len(js.name)
                    or not required.issubset(js.name) or len(js.velocity) != len(js.name)
                    or not all(math.isfinite(value) for value in js.position)):
                stable_since, reference = None, None
                continue
            joints = dict(zip(js.name, js.position))
            velocities = dict(zip(js.name, js.velocity))
            if any(not math.isfinite(velocities[n]) or abs(velocities[n]) > 0.005
                   for n in required):
                stable_since, reference = None, None
                continue
            now = time.monotonic()
            if reference is None or any(abs(joints[n] - reference[n]) > 0.0005
                                        for n in required):
                stable_since, reference = now, joints
            elif now - stable_since >= 0.3:
                return True
        return False

    def wait_for_servers(self):
        for client in (self.plan_client, self.validity_client,
                       self.scene_client, self.apply_client, self.limit_client):
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError("A MoveIt service is unavailable.")
        for side, client in zip(("left", "right"), self.exec_clients):
            if not client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"{side} arm controller is unavailable.")
            params = self.create_client(GetParameters,
                f"/{side}_fr3_arm_controller/get_parameters")
            try:
                if not params.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError("Cannot check controller interpolation.")
                req = GetParameters.Request()
                req.names = ["interpolation_method"]
                result = self.wait(params.call_async(req))
                if not result.values or result.values[0].string_value != "splines":
                    raise RuntimeError("Controller interpolation_method must be 'splines'.")
            finally:
                self.destroy_client(params)

    def configure_planner_sampling(self):
        # Every condition uses the same planner sampling setting.
        setter = self.create_client(SetPlannerParams, "/set_planner_params")
        getter = self.create_client(GetPlannerParams, "/get_planner_params")
        try:
            if not setter.wait_for_service(timeout_sec=10.0) or not getter.wait_for_service(timeout_sec=10.0):
                raise RuntimeError("OMPL planner-parameter services are unavailable.")
            for group in ("left_fr3_arm", "right_fr3_arm"):
                request = SetPlannerParams.Request()
                request.pipeline_id = "ompl"
                request.planner_config = "RRTConnectkConfigDefault"
                request.group = group
                request.replace = False
                request.params.keys = ["longest_valid_segment_fraction"]
                request.params.values = ["0.0005"]
                self.wait(setter.call_async(request))
                check = GetPlannerParams.Request()
                check.pipeline_id = request.pipeline_id
                check.planner_config = request.planner_config
                check.group = group
                params = self.wait(getter.call_async(check)).params
                values = dict(zip(params.keys, params.values))
                if abs(float(values.get("longest_valid_segment_fraction", "nan")) - 0.0005) > 1e-9 or not math.isfinite(float(values.get("longest_valid_segment_fraction", "nan"))):
                    raise RuntimeError(f"Could not verify OMPL collision sampling for {group}.")
        finally:
            self.destroy_client(setter)
            self.destroy_client(getter)

    def state_valid(self, state, constraints=None):
        req = GetStateValidity.Request()
        req.group_name = ""  # Collision check includes both arms and hands.
        req.robot_state = state
        if constraints is not None:
            req.constraints = constraints
        response = self.wait(self.validity_client.call_async(req), 10.0)
        return response.valid

    def pair_samples(self, trajectories, state):
        start = dict(zip(state.joint_state.name, state.joint_state.position))
        samples = [TrajectorySamples(t, start) for t in trajectories]
        times = sorted(set().union(*(s.check_times(self.check_dt, self.check_joint_step)
                                    for s in samples)))
        if len(times) > 30000:
            raise RuntimeError("Too many collision samples.")
        return samples, times

    def goals_reached(self, state, left, right):
        return sum(self.state_valid(state, self.build_joint_goal_constraints(target))
                   for target in (left, right))

    def build_start_state(self) -> RobotState:
        if not self.wait_for_joint_state():
            raise RuntimeError("No complete, stationary joint state received.")
        rs = RobotState()
        rs.joint_state = copy.deepcopy(self.latest_joint_state)
        rs.is_diff = True
        return rs

    def build_joint_goal_constraints(self, joint_map: dict) -> Constraints:
        c = Constraints()
        for jname, jval in joint_map.items():
            c.joint_constraints.append(JointConstraint(
                joint_name=jname, position=float(jval), weight=1.0,
                tolerance_above=0.001, tolerance_below=0.001))
        return c

    def plan_for_group(self, group_name: str, joint_map: dict, start_state):
        req = GetMotionPlan.Request()

        mpr = MotionPlanRequest()
        mpr.group_name = group_name
        mpr.num_planning_attempts = 5
        mpr.allowed_planning_time = 10.0
        mpr.pipeline_id = "ompl"
        mpr.planner_id = "RRTConnectkConfigDefault"
        mpr.max_velocity_scaling_factor = self.velocity_scale
        mpr.max_acceleration_scaling_factor = self.acceleration_scale

        # Both plans use same starting positions.
        mpr.start_state = copy.deepcopy(start_state)
        mpr.goal_constraints = [self.build_joint_goal_constraints(joint_map)]
        req.motion_plan_request = mpr

        future = self.plan_client.call_async(req)
        res = self.wait(future, 60.0).motion_plan_response
        if res.error_code.val != 1:
            raise RuntimeError(f"Planning failed for {group_name}, error_code={res.error_code.val}")

        traj = res.trajectory.joint_trajectory
        if (len(traj.joint_names) != 7 or set(traj.joint_names) != set(joint_map)
                or not traj.points):
            raise RuntimeError("Incomplete arm trajectory.")
        return traj

    def execute_trajectory(self, trajectories):
        margin = 0.1
        if not math.isfinite(self.start_delay) or self.start_delay <= margin:
            raise RuntimeError("Start delay must be greater than 0.1 seconds.")
        deadline = time.monotonic() + self.start_delay - margin
        start = self.get_clock().now() + Duration(seconds=self.start_delay)
        accept_before = start.nanoseconds - int(margin * 1e9)
        sends = []
        abort = False

        def late_response(future):
            try:
                handle = future.result()
                if abort and handle is not None and handle.accepted:
                    handle.cancel_goal_async()
            except Exception:
                self.get_logger().error("Could not cancel a late controller response.")

        try:
            for client, trajectory in zip(self.exec_clients, trajectories):
                goal = FollowJointTrajectory.Goal()
                goal.trajectory = copy.deepcopy(trajectory)
                # Both arms have same start time.
                goal.trajectory.header.stamp = start.to_msg()
                goal.goal_time_tolerance = Duration(seconds=2.0).to_msg()
                for name in trajectory.joint_names:
                    goal.path_tolerance.append(JointTolerance(name=name, position=0.05))
                    goal.goal_tolerance.append(JointTolerance(name=name, position=0.01))
                future = client.send_goal_async(goal)
                sends.append(future)
                future.add_done_callback(late_response)
            while not all(f.done() for f in sends):
                if (not rclpy.ok() or time.monotonic() >= deadline or
                        self.get_clock().now().nanoseconds >= accept_before):
                    raise RuntimeError("Both controllers did not accept before the start deadline.")
                rclpy.spin_once(self, timeout_sec=0.01)
            handles = [f.result() for f in sends]
            if any(h is None or not h.accepted for h in handles):
                raise RuntimeError("An arm controller rejected its trajectory.")
            if (time.monotonic() >= deadline or
                    self.get_clock().now().nanoseconds >= accept_before):
                raise RuntimeError("Too little time remains before the common start.")
            results = [h.get_result_async() for h in handles]
            duration = max(seconds(t.points[-1].time_from_start) for t in trajectories)
            deadline = time.monotonic() + 5 * (duration + self.start_delay) + 30
            while True:
                if not rclpy.ok() or time.monotonic() > deadline:
                    raise RuntimeError("Arm execution timed out.")
                if self.get_clock().now().nanoseconds < start.nanoseconds:
                    js = self.latest_joint_state
                    if js is not None:
                        actual = dict(zip(js.name, js.position))
                        for trajectory in trajectories:
                            for name, position in zip(trajectory.joint_names,
                                                      trajectory.points[0].positions):
                                if name not in actual or abs(actual[name] - position) > 0.005:
                                    raise RuntimeError("Arm moved before the scheduled start.")
                for f in results:
                    if f.done():
                        result = f.result()
                        if (result is None or result.status != GoalStatus.STATUS_SUCCEEDED
                                or result.result.error_code != 0):
                            raise RuntimeError(f"Arm execution failed: {result}")
                if all(f.done() for f in results):
                    return True
                rclpy.spin_once(self, timeout_sec=0.02)
        except BaseException:
            # Both arms are stopped if something fails.
            abort = True
            for future in sends:
                if future.done() and future.exception() is None:
                    handle = future.result()
                    if handle is not None and handle.accepted:
                        handle.cancel_goal_async()
            deadline = time.monotonic() + 3.0
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            self.get_logger().error("Stop requested for both arms. Check controller state.")
            raise

    def move_home(self):
        state = self.build_start_state()
        joints = dict(zip(state.joint_state.name, state.joint_state.position))
        target = {**self.left_home, **self.right_home}
        if all(abs(joints[name] - value) < 0.001 for name, value in target.items()):
            return True
        return self.move_pair(self.left_home, self.right_home)["success"]

    def run(self):
        self.wait_for_servers()

        self.load_joint_limits()
        self.configure_planner_sampling()
        self.enable_cross_arm_checks()

        if not self.move_home():
            self.get_logger().error("Initial HOME failed.")
            return

        # The left arm finishes its three poses before right arm.
        movements = [
            ("LEFT", "pose1", self.left_pose1),
            ("LEFT", "pose2", self.left_pose2),
            ("LEFT", "pose3", self.left_pose3),
            ("RIGHT", "pose4", self.right_pose4),
            ("RIGHT", "pose5", self.right_pose5),
            ("RIGHT", "pose6", self.right_pose6),
        ]

        results = []
        total_start = time.perf_counter()

        for arm, pose_name, target in movements:
            side = arm.lower()
            result = self.move_pair(
                target if side == "left" else self.left_home,
                target if side == "right" else self.right_home,
                active_side=side)
            results.append(result)

            if not result["success"]:
                self.get_logger().error(f"{pose_name} failed. Stopping sequence.")
                break

        # Task time excludes return HOME.
        total_time = time.perf_counter() - total_start
        complete_success = (len(results) == len(movements)
                            and all(r["success"] for r in results))

        # Both arms return HOME after all six target moves succeed.
        try:
            if complete_success:
                if not self.move_home():
                    self.get_logger().error("Final HOME failed (outside the timed task).")
            else:
                self.get_logger().warning("Experiment incomplete. Final HOME skipped.")
        finally:
            self.print_trial_results(results, total_time, complete_success)

    def print_trial_results(self, results, total_time, success):
        def sum_field(field):
            values = [r[field] for r in results if r[field] is not None]
            return sum(values) if values else None

        print_trial_table(
            self, sum_field("planning_time"), sum_field("validation_time"),
            sum_field("execution_time"), total_time, success,
            sum(r["successful_poses"] for r in results)
        )

    def wait(self, future, timeout=15.0):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError("ROS request timed out or returned no result.")
        return future.result()

    def move_pair(self, left_target, right_target, active_side=None):
        result = dict(planning_time=None, validation_time=None, execution_time=None,
                      successful_poses=0, success=False, status="SETUP FAILED")
        try:
            start_state = self.build_start_state()
            scene_key = self.scene_key(self.get_scene())
            # The inactive arm keeps its measured position, not its HOME target.
            positions = dict(zip(start_state.joint_state.name, start_state.joint_state.position))
            if active_side == "left":
                right_target = {name: positions[name] for name in self.right_home}
            elif active_side == "right":
                left_target = {name: positions[name] for name in self.left_home}
            elif active_side is not None:
                raise ValueError("Active side must be left, right or None.")
            result["status"] = "PLANNING FAILED"
            begin = time.perf_counter()
            try:
                trajectories = [
                    self.plan_for_group(f"{side}_fr3_arm", target, start_state)
                    for side, target in (("left", left_target), ("right", right_target))
                    if active_side is None or side == active_side]
            finally:
                result["planning_time"] = time.perf_counter() - begin

            result["status"] = "PREPARATION FAILED"
            trajectories = [self.prepare_trajectory(t, start_state) for t in trajectories]
            if active_side is not None:
                # The other arm holds its measured position for the whole move.
                moving = trajectories[0]
                joints = self.right_home if active_side == "left" else self.left_home
                hold = copy.deepcopy(moving)
                hold.joint_names = list(joints)
                for point in hold.points:
                    point.positions = [positions[name] for name in joints]
                    point.velocities = [0.0] * 7
                    point.accelerations = [0.0] * 7
                    point.effort = []
                trajectories = ([moving, hold] if active_side == "left"
                                else [hold, moving])

            result["status"] = "COLLISION CHECK FAILED"
            begin = time.perf_counter()
            try:
                # Checking includes spline samples and endpoint goal checks.
                samples, times = self.pair_samples(trajectories, start_state)
                names = [name for traj in trajectories for name in traj.joint_names]
                points = [[q for sample in samples for q in sample.positions(t)] for t in times]
                valid_pair, state = self.validate_pair(points, names, start_state)
                if not valid_pair:
                    raise RuntimeError("The planned pair failed collision checking.")
                if self.goals_reached(state, left_target, right_target) != 2:
                    raise RuntimeError("Planned endpoints do not satisfy both targets.")
            finally:
                result["validation_time"] = time.perf_counter() - begin

            result["status"] = "PRE-EXECUTION CHECK FAILED"
            if self.scene_key(self.get_scene()) != scene_key:
                raise RuntimeError("Planning scene changed; pair will not execute.")
            self.check_start_unchanged(start_state)
            result["status"] = "EXECUTION FAILED"
            begin = time.perf_counter()
            try:
                self.execute_trajectory(trajectories)
            finally:
                result["execution_time"] = time.perf_counter() - begin
            result["status"] = "GOAL CHECK FAILED"
            result["successful_poses"] = self.goals_reached(self.build_start_state(), left_target, right_target)
            result["success"] = result["successful_poses"] == 2
            if active_side is not None:
                # Holding other arm does not count as a target pose.
                result["successful_poses"] = int(result["success"])
            result["status"] = "SUCCESS" if result["success"] else "GOAL CHECK FAILED"
        except Exception as error:
            self.get_logger().error(f"{result['status']}: {error}")
        return result

    def load_joint_limits(self):
        names = list(self.left_home) + list(self.right_home)
        fields = ("has_velocity_limits", "max_velocity",
                  "has_acceleration_limits", "max_acceleration")
        req = GetParameters.Request()
        req.names = [f"robot_description_planning.joint_limits.{name}.{field}"
                     for name in names for field in fields]
        values = self.wait(self.limit_client.call_async(req)).values
        if len(values) != len(req.names):
            raise RuntimeError("MoveIt did not return all joint-limit parameters.")
        for i, name in enumerate(names):
            has_v, max_v, has_a, max_a = values[4*i:4*i+4]
            if (has_v.type != ParameterType.PARAMETER_BOOL or not has_v.bool_value
                    or has_a.type != ParameterType.PARAMETER_BOOL or not has_a.bool_value):
                raise RuntimeError(f"Velocity/acceleration limits unavailable for {name}.")
            limits = []
            for value in (max_v, max_a):
                if value.type == ParameterType.PARAMETER_DOUBLE:
                    number = value.double_value
                elif value.type == ParameterType.PARAMETER_INTEGER:
                    number = float(value.integer_value)
                else:
                    raise RuntimeError(f"Missing numeric joint limit for {name}.")
                if not math.isfinite(number) or number <= 0:
                    raise RuntimeError(f"Invalid joint limit for {name}.")
                limits.append(number)
            self.joint_limits[name] = (limits[0] * self.velocity_scale,
                                       limits[1] * self.acceleration_scale)

    def prepare_trajectory(self, trajectory, start_state):
        start = dict(zip(start_state.joint_state.name, start_state.joint_state.position))
        # Planner path is checked before its timing is changed.
        TrajectorySamples(trajectory, start, check_finish=False)
        traj = copy.deepcopy(trajectory)
        n = len(traj.joint_names)
        if any(len(point.velocities) != n for point in traj.points):
            raise RuntimeError("Timing preparation requires waypoint velocities.")
        for point in (traj.points[0], traj.points[-1]):
            point.velocities = [0.0] * n
            point.accelerations = [0.0] * n

        # Joint limits are checked after endpoints are set.
        scale = self.required_time_scale(traj, start)
        if scale > 1.0:
            scale *= 1.001  # Margin for rounding.
            for point in traj.points:

                point.time_from_start = Duration(
                    seconds=seconds(point.time_from_start) * scale).to_msg()
                point.velocities = [v / scale for v in point.velocities]
                point.accelerations = [a / scale**2 for a in point.accelerations]
        if self.required_time_scale(traj, start) > 1.00001:
            raise RuntimeError("Prepared spline exceeds joint limits.")
        return traj

    def required_time_scale(self, trajectory, start):
        samples = TrajectorySamples(trajectory, start)
        scale = 1.0
        for i, segment in enumerate(samples.segments):
            dt = samples.times[i+1] - samples.times[i]
            for name, coefficients in zip(trajectory.joint_names, segment):
                max_v, max_a = self.joint_limits[name]
                velocity = [k*c / dt for k, c in enumerate(coefficients) if k]
                acceleration = [k*c / dt for k, c in enumerate(velocity) if k]
                scale = max(scale, polynomial_bound(velocity) / max_v,
                            math.sqrt(polynomial_bound(acceleration) / max_a))
        return scale


    def validate_pair(self, points, names, start_state):
        # Both arm positions are checked at matching times.
        for values in points:
            state = copy.deepcopy(start_state)
            joints = dict(zip(state.joint_state.name, state.joint_state.position))
            joints.update(zip(names, values))
            state.joint_state.position = [joints[n] for n in state.joint_state.name]
            if not self.state_valid(state):
                return False, state
        return True, state

    def check_start_unchanged(self, start_state):
        current = self.build_start_state()
        joints = dict(zip(current.joint_state.name, current.joint_state.position))
        for name, value in zip(start_state.joint_state.name, start_state.joint_state.position):
            if name not in joints or abs(joints[name] - value) > 0.001:
                raise RuntimeError("Robot moved after planning. Pair will not execute.")

    def get_scene(self):
        req = GetPlanningScene.Request()
        req.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
            PlanningSceneComponents.OCTOMAP |
            PlanningSceneComponents.TRANSFORMS |
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX |
            PlanningSceneComponents.LINK_PADDING_AND_SCALING |
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        return self.wait(self.scene_client.call_async(req)).scene

    def scene_key(self, scene):
        transforms = copy.deepcopy(scene.fixed_frame_transforms)
        for transform in transforms:
            transform.header.stamp.sec = 0
            transform.header.stamp.nanosec = 0
        return repr((scene.world, transforms,
                     scene.allowed_collision_matrix, scene.link_padding,
                     scene.link_scale, scene.robot_state.attached_collision_objects))

    def enable_cross_arm_checks(self):
        # Collision checks between arms remain enabled.
        scene = self.get_scene()
        acm = scene.allowed_collision_matrix
        left = [i for i, name in enumerate(acm.entry_names) if name.startswith("left_")]
        right = [i for i, name in enumerate(acm.entry_names) if name.startswith("right_")]
        if not left or not right:
            raise RuntimeError("Cannot identify both arms in the collision matrix.")
        changed = 0
        for i in left:
            for j in right:
                if acm.entry_values[i].enabled[j] or acm.entry_values[j].enabled[i]:
                    changed += 1
                acm.entry_values[i].enabled[j] = False
                acm.entry_values[j].enabled[i] = False
        if changed:
            req = ApplyPlanningScene.Request()
            req.scene.is_diff = True
            req.scene.robot_state.is_diff = True
            req.scene.allowed_collision_matrix = acm
            if not self.wait(self.apply_client.call_async(req)).success:
                raise RuntimeError("Could not enable cross-arm collision checks.")
            self.get_logger().info(f"Enabled {changed} cross-arm collision checks.")
        actual = self.get_scene().allowed_collision_matrix
        if list(actual.entry_names) != list(acm.entry_names):
            raise RuntimeError("Collision matrix changed while enabling checks.")
        if any(actual.entry_values[i].enabled[j] or actual.entry_values[j].enabled[i]
               for i in left for j in right):
            raise RuntimeError("Cross-arm collision checks are still disabled.")


def seconds(duration):
    return duration.sec + duration.nanosec * 1e-9


def polynomial_bound(coefficients):
    # Bernstein coefficients bound values between path points.
    degree = len(coefficients) - 1
    if degree < 0:
        return 0.0
    values = [sum(coefficients[i] * math.comb(k, i) / math.comb(degree, i)
                  for i in range(k+1)) for k in range(degree+1)]
    return max(abs(value) for value in values)


def spline_coefficients(p0, p1, dt):
    # Coefficients describe curve between two path points.
    coefficients = []
    for j, start in enumerate(p0.positions):
        distance = p1.positions[j] - start
        if p0.velocities and p1.velocities:
            v0, v1 = p0.velocities[j] * dt, p1.velocities[j] * dt
            if p0.accelerations and p1.accelerations:
                a0 = p0.accelerations[j] * dt * dt
                a1 = p1.accelerations[j] * dt * dt
                coefficients.append([
                    start, v0, a0 / 2,
                    10*distance - 6*v0 - 4*v1 - 1.5*a0 + 0.5*a1,
                    -15*distance + 8*v0 + 7*v1 + 1.5*a0 - a1,
                    6*distance - 3*v0 - 3*v1 - 0.5*a0 + 0.5*a1])
            else:
                coefficients.append([start, v0,
                                     3*distance - 2*v0 - v1,
                                     -2*distance + v0 + v1])
        else:
            coefficients.append([start, distance])
    return coefficients


class TrajectorySamples:
    def __init__(self, trajectory, start_state, check_finish=True):
        self.trajectory = trajectory
        self.times = [seconds(p.time_from_start) for p in trajectory.points]
        self.segments = []
        n = len(trajectory.joint_names)
        if not self.times or abs(self.times[0]) > 1e-9:
            raise RuntimeError("Trajectory must start at time zero.")
        for point in trajectory.points:
            if len(point.positions) != n:
                raise RuntimeError("Missing trajectory positions.")
            for values in (point.positions, point.velocities, point.accelerations):
                if values and (len(values) != n or not all(math.isfinite(v) for v in values)):
                    raise RuntimeError("Invalid trajectory values.")
            if point.accelerations and not point.velocities:
                raise RuntimeError("Accelerations require velocities.")
        for name, value in zip(trajectory.joint_names, trajectory.points[0].positions):
            if abs(value - start_state[name]) > 0.001:
                raise RuntimeError("Planner changed the start state. Stop and check it.")
        # Arm stays still after its path ends.
        point = trajectory.points[-1]
        if check_finish and any(abs(v) > 1e-5 for v in point.velocities):
            raise RuntimeError(
                f"{trajectory.joint_names[0]}: finish velocities = "
                f"{list(point.velocities)}"
            )
        for i in range(len(self.times) - 1):
            dt = self.times[i+1] - self.times[i]
            if dt <= 0:
                raise RuntimeError("Trajectory times must increase.")
            self.segments.append(spline_coefficients(
                trajectory.points[i], trajectory.points[i+1], dt))

    def positions(self, t):
        if t >= self.times[-1]:
            return list(self.trajectory.points[-1].positions)
        i = max(0, bisect.bisect_right(self.times, t) - 1)
        u = (t - self.times[i]) / (self.times[i+1] - self.times[i])
        return [sum(c * u**k for k, c in enumerate(poly))
                for poly in self.segments[i]]

    def check_times(self, time_step, joint_step):
        times = set(self.times)
        for i, coefficients in enumerate(self.segments):
            dt = self.times[i+1] - self.times[i]
            bound = max(sum(k * abs(c) for k, c in enumerate(poly))
                        for poly in coefficients)
            count = max(1, math.ceil(dt / time_step), math.ceil(bound / joint_step))
            if count > 30000:
                raise RuntimeError("Trajectory needs too many collision samples.")
            times.update(self.times[i] + dt*j/count for j in range(count+1))
        return times


def print_trial_table(node, planning, checking, execution, total, success, poses):
    def seconds(value):
        return "N/A" if value is None else f"{value:.4f} s"

    rows = [
        ("Planning time", seconds(planning)),
        ("Collision-checking time", seconds(checking)),
        ("Execution time", seconds(execution)),
        ("Total completion time", seconds(total)),
        ("Task success", "SUCCESS" if success else "FAILED"),
        ("Successful poses", f"{poses}/6"),
    ]
    lines = ["", "================ TRIAL RESULTS ================",
             f"{'Measurement':<27} {'Result':>14}", "-" * 46]
    lines.extend(f"{label:<27} {value:>14}" for label, value in rows)
    lines.append("=" * 46)
    node.get_logger().info("\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DualArmMoveGroupJointPose()
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().error(str(error))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()