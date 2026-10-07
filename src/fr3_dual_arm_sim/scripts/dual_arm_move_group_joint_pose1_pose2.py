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
        super().__init__("dual_arm_move_group_joint_pose1_pose2")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.latest_joint_state = None
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.exec_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

        # ---Target poses---
        self.right_pose1 = {
            "right_fr3_joint1": 0.0,
            "right_fr3_joint2": -0.5,
            "right_fr3_joint3": 0.0,
            "right_fr3_joint4": -1.5,
            "right_fr3_joint5": 0.0,
            "right_fr3_joint6": 1.0,
            "right_fr3_joint7": 0.5,
        }

        self.left_pose2 = {
            "left_fr3_joint1": 0.0,
            "left_fr3_joint2": -0.5,
            "left_fr3_joint3": 0.0,
            "left_fr3_joint4": -1.5,
            "left_fr3_joint5": 0.0,
            "left_fr3_joint6": 1.0,
            "left_fr3_joint7": 0.5,
        }

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

        # 1) Right arm -> pose1
        self.get_logger().info("Planning RIGHT arm to pose1...")
        traj_r = self.plan_for_group("right_fr3_arm", self.right_pose1)
        if traj_r:
            self.get_logger().info("Executing RIGHT arm...")
            self.execute_trajectory(traj_r)

        # 2) Left arm -> pose2
        self.get_logger().info("Planning LEFT arm to pose2...")
        traj_l = self.plan_for_group("left_fr3_arm", self.left_pose2)
        if traj_l:
            self.get_logger().info("Executing LEFT arm...")
            self.execute_trajectory(traj_l)


def main(args=None):
    rclpy.init(args=args)
    node = DualArmMoveGroupJointPose()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

