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
        # --- Target poses ---
        base = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]

        def pose(arm, joint1, joint3):
            joints = base.copy()
            joints[0] = joint1
            joints[2] = joint3
            return {
                f"{arm}_fr3_joint{i}": value
                for i, value in enumerate(joints, start=1)
            }

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
        mpr.allowed_planning_time = 5.0

        mpr.pipeline_id = "ompl"
        mpr.planner_id = ""  

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

    def run(self):
        if not self.wait_for_servers():
            return
        if not self.wait_for_joint_state():
            self.get_logger().error("No /joint_states received. Is simulation running?")
            return

        # 1) Left arm -> pose1
        self.get_logger().info("Planning LEFT arm to pose1...")
        traj_l1 = self.plan_for_group("left_fr3_arm", self.left_pose1)
        if traj_l1:
            self.get_logger().info("Executing LEFT arm to pose1...")
            self.execute_trajectory(traj_l1)

        # 2) Left arm -> pose2
        self.get_logger().info("Planning LEFT arm to pose2...")
        traj_l2 = self.plan_for_group("left_fr3_arm", self.left_pose2)
        if traj_l2:
            self.get_logger().info("Executing LEFT arm to pose2...")
            self.execute_trajectory(traj_l2)

        # 3) Left arm -> pose3
        self.get_logger().info("Planning LEFT arm to pose3...")
        traj_l3 = self.plan_for_group("left_fr3_arm", self.left_pose3)
        if traj_l3:
            self.get_logger().info("Executing LEFT arm to pose3...")
            self.execute_trajectory(traj_l3)

        # 4) Right arm -> pose4
        self.get_logger().info("Planning RIGHT arm to pose4...")
        traj_r4 = self.plan_for_group("right_fr3_arm", self.right_pose4)
        if traj_r4:
            self.get_logger().info("Executing RIGHT arm to pose4...")
            self.execute_trajectory(traj_r4)

        # 5) Right arm -> pose5
        self.get_logger().info("Planning RIGHT arm to pose5...")
        traj_r5 = self.plan_for_group("right_fr3_arm", self.right_pose5)
        if traj_r5:
            self.get_logger().info("Executing RIGHT arm to pose5...")
            self.execute_trajectory(traj_r5)

        # 6) Right arm -> pose6
        self.get_logger().info("Planning RIGHT arm to pose6...")
        traj_r6 = self.plan_for_group("right_fr3_arm", self.right_pose6)
        if traj_r6:
            self.get_logger().info("Executing RIGHT arm to pose6...")
            self.execute_trajectory(traj_r6)

        
def main(args=None):
    rclpy.init(args=args)
    node = DualArmMoveGroupJointPose()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

