#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.action import ActionClient

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.action import ExecuteTrajectory
from control_msgs.action import GripperCommand
from action_msgs.msg import GoalStatus


class DualArmMovement(Node):

    def __init__(self):
        super().__init__("both_arms_gripper_twice")
        self.set_parameters([
            Parameter("use_sim_time", Parameter.Type.BOOL, True)
        ])

        self.plan_client = self.create_client(
            GetMotionPlan, "/plan_kinematic_path"
        )
        self.exec_client = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory"
        )

        self.grippers = [
            ActionClient(
                self, GripperCommand,
                f"/{side}_fr3_hand_controller/gripper_cmd"
            )
            for side in ("left", "right")
        ]

    def wait(self, future):
        rclpy.spin_until_future_complete(self, future)
        if not future.done():
            raise RuntimeError("ROS stopped while waiting.")
        return future.result()

    def move_arms(self, left_joint1, right_joint1):
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request

        motion.group_name = "both_arms"
        motion.pipeline_id = "ompl"
        motion.planner_id = "RRTConnectkConfigDefault"
        motion.num_planning_attempts = 5
        motion.allowed_planning_time = 10.0
        motion.max_velocity_scaling_factor = 0.2
        motion.max_acceleration_scaling_factor = 0.2

        # Use the current robot state from MoveIt.
        motion.start_state.is_diff = True

        constraints = Constraints()

        # Same joint targets as your original script.
        for side, joint1 in (
            ("left", left_joint1),
            ("right", right_joint1),
        ):
            positions = [joint1, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]

            for number, position in enumerate(positions, start=1):
                joint = JointConstraint()
                joint.joint_name = f"{side}_fr3_joint{number}"
                joint.position = position
                joint.tolerance_above = 0.001
                joint.tolerance_below = 0.001
                joint.weight = 1.0
                constraints.joint_constraints.append(joint)

        motion.goal_constraints = [constraints]

        response = self.wait(self.plan_client.call_async(request))
        plan = response.motion_plan_response

        if plan.error_code.val != 1:
            raise RuntimeError(
                f"Planning failed: {plan.error_code.val}"
            )

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan.trajectory

        handle = self.wait(self.exec_client.send_goal_async(goal))

        if not handle.accepted:
            raise RuntimeError("Arm movement was rejected.")

        result = self.wait(handle.get_result_async())

        if (
            result.status != GoalStatus.STATUS_SUCCEEDED
            or result.result.error_code.val != 1
        ):
            raise RuntimeError("Arm movement failed.")

    def move_grippers(self, position):
        futures = []

        # Send both commands before waiting.
        for client in self.grippers:
            goal = GripperCommand.Goal()
            goal.command.position = position
            goal.command.max_effort = 10.0
            futures.append(client.send_goal_async(goal))

        handles = [self.wait(future) for future in futures]

        if not all(handle.accepted for handle in handles):
            for handle in handles:
                if handle.accepted:
                    self.wait(handle.cancel_goal_async())
            raise RuntimeError("A gripper command was rejected.")

        results = [
            handle.get_result_async()
            for handle in handles
        ]

        for future in results:
            result = self.wait(future)

            if (
                result.status != GoalStatus.STATUS_SUCCEEDED
                or not result.result.reached_goal
            ):
                raise RuntimeError("A gripper did not reach its target.")

    def run(self):
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("Planning service is unavailable.")

        for client in [self.exec_client] + self.grippers:
            if not client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError("An action server is unavailable.")
        # Move the fingers away from the zero limit before planning.
        self.get_logger().info("Opening grippers before arm movement")
        self.move_grippers(0.035)

        # Allow MoveIt to receive the updated joint positions.
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Both arms: first position")
        self.move_arms(0.0, 0.0)

        self.get_logger().info("Both arms: second position")
        self.move_arms(-0.8, 0.8)

        for cycle in range(1, 3):
            self.get_logger().info(f"Cycle {cycle}: open")
            self.move_grippers(0.035)

            self.get_logger().info(f"Cycle {cycle}: close")
            self.move_grippers(0.0)

        self.get_logger().info("Finished.")


def main():
    rclpy.init()
    node = DualArmMovement()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        node.get_logger().error(str(error))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()