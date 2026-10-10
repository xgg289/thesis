#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import MotionPlanRequest, Constraints, JointConstraint, RobotState
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient


class DualArmMoveGroupJointPose(Node):
    """
    Plans using the existing /move_group (GetMotionPlan service) and executes using /execute_trajectory action.
    This avoids MoveItPy pipeline-loading problems in a separate Python process.
    """

    def __init__(self):
        super().__init__("sequential_dual_arm_6poses")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.latest_joint_state = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.exec_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

        # ---Target poses---
        base = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]

        def pose(arm, joint1, joint3):
            joints = base.copy()
            joints[0] = joint1
            joints[2] = joint3
            return {
                f"{arm}_fr3_joint{i}": value
                for i, value in enumerate(joints, start=1)
            }
        #Home
        self.left_home = pose("left", 0.0, 0.0)
        self.right_home = pose("right", 0.0, 0.0)

        # Left arm: pose1, pose2, pose3
        self.left_pose1 = pose("left", -0.8, 0.3)
        self.left_pose2 = pose("left",  0.0, -0.4)
        self.left_pose3 = pose("left",  0.8, 0.3)

        # Right arm: pose4, pose5, pose6
        self.right_pose4 = pose("right",  0.8, -0.3)
        self.right_pose5 = pose("right",  0.0, 0.4)
        self.right_pose6 = pose("right", -0.8, -0.3)


    def _js_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_joint_state(self, timeout_sec=10.0) -> bool:
        t0 = time.time()
        while rclpy.ok() and self.latest_joint_state is None and (time.time() - t0) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.latest_joint_state is not None

    def wait_for_servers(self) -> bool:
        self.get_logger().info("Waiting for /plan_kinematic_path service...")
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/plan_kinematic_path service not available.")
            return False

        self.get_logger().info("Waiting for /execute_trajectory action server...")
        if not self.exec_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("/execute_trajectory action server not available.")
            return False

        return True

    def build_start_state(self) -> RobotState:
        js = self.latest_joint_state
        rs = RobotState()
        rs.joint_state.name = list(js.name)
        rs.joint_state.position = list(js.position)
        return rs

    def build_joint_goal_constraints(self, joint_map: dict) -> Constraints:
        c = Constraints()
        for jname, jval in joint_map.items():
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(jval)
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    def plan_for_group(self, group_name: str, joint_map: dict):
        req = GetMotionPlan.Request()

        mpr = MotionPlanRequest()
        mpr.group_name = group_name
        mpr.num_planning_attempts = 5
        mpr.allowed_planning_time = 10.0

        mpr.pipeline_id = "ompl"
        mpr.planner_id = "RRTConnectkConfigDefault"  
        mpr.max_velocity_scaling_factor = 1.0
        mpr.max_acceleration_scaling_factor = 1.0

        mpr.start_state = self.build_start_state()
        mpr.goal_constraints = [self.build_joint_goal_constraints(joint_map)]

        req.motion_plan_request = mpr

        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        if not future.done() or future.result() is None:
            self.get_logger().error(f"Planning service call failed for group {group_name}.")
            return None

        res = future.result().motion_plan_response
        if res.error_code.val != 1:  
            self.get_logger().error(f"Planning failed for {group_name}, error_code={res.error_code.val}")
            return None

        self.get_logger().info(f"Planning succeeded for {group_name}.")
        return res.trajectory

    def execute_trajectory(self, traj) -> bool:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.exec_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("ExecuteTrajectory goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        result = result_future.result()
        if result is None:
            self.get_logger().error("No result from ExecuteTrajectory.")
            return False

        
        ok = (result.result.error_code.val == 1)
        if ok:
            self.get_logger().info("Execution SUCCESS ✅")
        else:
            self.get_logger().error(f"Execution FAILED ❌ error_code={result.result.error_code.val}")
        return ok

    def move_home(self):
        for arm in ("left", "right"):
            rclpy.spin_once(self, timeout_sec=0.1)
            target = getattr(self, f"{arm}_home")
            joints = dict(zip(self.latest_joint_state.name,
                              self.latest_joint_state.position))

            if all(abs(joints.get(j, float("inf")) - v) < 0.03
                   for j, v in target.items()):
                continue

            traj = self.plan_for_group(f"{arm}_fr3_arm", target)
            if traj is None or not self.execute_trajectory(traj):
                return False

        return True
       

    def run(self):
        if not self.wait_for_servers():
            return

        if not self.wait_for_joint_state():
            self.get_logger().error("No /joint_states received.")
            return
        
        # Initial HOME
        self.get_logger().info("Preparing initial HOME position...")

        if not self.move_home():
            self.get_logger().error("Initial HOME failed.")
            return

        # Sequential movement: LEFT -> RIGHT
        movements = [
            ("LEFT", "pose1", "left_fr3_arm", self.left_pose1),
            ("RIGHT", "pose4", "right_fr3_arm", self.right_pose4),
            ("LEFT", "pose2", "left_fr3_arm", self.left_pose2),
            ("RIGHT", "pose5", "right_fr3_arm", self.right_pose5),
            ("LEFT", "pose3", "left_fr3_arm", self.left_pose3),
            ("RIGHT", "pose6", "right_fr3_arm", self.right_pose6),
        ]

        results = []
        total_start = time.perf_counter()

        for arm, pose_name, group, target in movements:

            self.get_logger().info(
                f"Planning {arm} {pose_name}..."
            )

            # Planning time
            planning_start = time.perf_counter()

            trajectory = None
            planning_error = None

            try:
                trajectory = self.plan_for_group(group, target)
            except Exception as exc:
                planning_error = str(exc)

            planning_time = time.perf_counter() - planning_start

            # Execution results
            execution_time = None
            success = False
            status = "PLANNING FAILED"

            if planning_error is not None:
                self.get_logger().error(
                    f"Planning error: {planning_error}"
                )

            if trajectory is not None:
                self.get_logger().info(
                    f"Executing {arm} {pose_name}..."
                )

                # Execution time
                execution_start = time.perf_counter()

                try:
                    success = self.execute_trajectory(trajectory)
                    status = "SUCCESS" if success else "EXECUTION FAILED"

                except Exception as exc:
                    success = False
                    status = "EXECUTION ERROR"
                    self.get_logger().error(
                        f"Execution error: {exc}"
                    )

                execution_time = (
                    time.perf_counter() - execution_start
                )

            results.append({
                "arm": arm,
                "pose": pose_name,
                "planning_time": planning_time,
                "execution_time": execution_time,
                "success": success,
                "status": status,
            })

            if not success:
                self.get_logger().error(
                    f"{arm} {pose_name} failed. Stopping sequence."
                )
                break

        # Total time
        total_time = time.perf_counter() - total_start

        # Planning and execution totals
        total_planning = sum(
            r["planning_time"] for r in results
        )

        left_time = sum(
            r["execution_time"]
            for r in results
            if r["arm"] == "LEFT"
            and r["execution_time"] is not None
        )

        right_time = sum(
            r["execution_time"]
            for r in results
            if r["arm"] == "RIGHT"
            and r["execution_time"] is not None
        )

        total_execution = left_time + right_time

        overhead = total_time - total_planning - total_execution

        successful_poses = sum(
            1 for r in results if r["success"]
        )

        complete_success = (
            len(results) == len(movements)
            and all(r["success"] for r in results)
        )

        # Terminal results table
        lines = [
            "",
            "================ INDIVIDUAL POSE RESULTS ================",
            f"{'ARM':<8} {'POSE':<8} {'PLAN (s)':>11} "
            f"{'EXEC (s)':>11} {'STATUS':>18}",
            "-" * 60,
        ]

        for r in results:
            execution = (
                f"{r['execution_time']:.4f}"
                if r["execution_time"] is not None
                else "N/A"
            )

            lines.append(
                f"{r['arm']:<8} "
                f"{r['pose']:<8} "
                f"{r['planning_time']:>11.4f} "
                f"{execution:>11} "
                f"{r['status']:>18}"
            )

        lines.extend([
            "-" * 60,
            "",
            "================ FINAL EXPERIMENT SUMMARY ===============",
            f"Total planning time:        {total_planning:.4f} s",
            f"Left-arm movement time:     {left_time:.4f} s",
            f"Right-arm movement time:    {right_time:.4f} s",
            f"Total execution time:       {total_execution:.4f} s",
            f"Other overhead:             {overhead:.4f} s",
            f"Total trial time:           {total_time:.4f} s",
            f"Successful poses:           {successful_poses}/{len(movements)}",
            f"Complete sequence success:  {complete_success}",
            "Collision-free execution:  NOT VERIFIED",
            "=========================================================",
        ])

        self.get_logger().info("\n".join(lines))


        # Return HOME 
        if complete_success:
            self.get_logger().info(
                "Six poses complete. Returning both arms HOME..."
            )

            if self.move_home():
                self.get_logger().info(
                    "Both arms returned HOME successfully."
                )
            else:
                self.get_logger().error("Final HOME failed.")
        else:
            self.get_logger().warning(
                "Experiment incomplete. Final HOME skipped."
            )




def main(args=None):
        rclpy.init(args=args)
        node = None

        try:
            node = DualArmMoveGroupJointPose()
            node.run()

        finally:
            if node is not None:
                node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
        main()
