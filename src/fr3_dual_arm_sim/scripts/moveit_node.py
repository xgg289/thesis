#!/usr/bin/env python3
"""Home both arms through the running MoveGroup action server."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node


class DualArmHoming(Node):
    def __init__(self):
        super().__init__('dual_arm_homing_node')
        self.move_group = ActionClient(self, MoveGroup, '/move_action')
        srdf = Path(get_package_share_directory('fr3_dual_arm_moveit_config')) / 'config/dual_fr3.srdf'
        self.home_states = {
            state.attrib['group']: state
            for state in ET.parse(srdf).getroot().findall('group_state')
            if state.attrib['name'] == 'home'
        }

    def go_home(self):
        if not self.move_group.wait_for_server(timeout_sec=60.0):
            self.get_logger().error('MoveGroup action server did not start')
            return False
        for side in ('left', 'right'):
            group = f'{side}_fr3_arm'
            controller = ActionClient(self, FollowJointTrajectory,
                                      f'/{side}_fr3_arm_controller/follow_joint_trajectory')
            ready = controller.wait_for_server(timeout_sec=60.0)
            controller.destroy()
            if not ready:
                self.get_logger().error(f'{side} arm controller did not start')
                return False
            goal = MoveGroup.Goal()
            goal.request.group_name = group
            goal.request.pipeline_id = 'ompl'
            goal.request.planner_id = 'RRTConnectkConfigDefault'
            goal.request.allowed_planning_time = 10.0
            goal.request.num_planning_attempts = 5
            goal.request.max_velocity_scaling_factor = 0.1
            goal.request.max_acceleration_scaling_factor = 0.1
            goal.request.start_state.is_diff = True
            constraints = Constraints(name='home')
            for joint in self.home_states[group].findall('joint'):
                constraints.joint_constraints.append(JointConstraint(
                    joint_name=joint.attrib['name'], position=float(joint.attrib['value']),
                    tolerance_above=0.001, tolerance_below=0.001, weight=1.0))
            goal.request.goal_constraints = [constraints]
            goal.planning_options.planning_scene_diff.is_diff = True
            goal.planning_options.planning_scene_diff.robot_state.is_diff = True
            self.get_logger().info(f'Planning and executing home for {side} arm...')
            future = self.move_group.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
            if not future.done() or not future.result().accepted:
                self.get_logger().error(f'Home goal was not accepted for {side} arm')
                return False
            result = future.result().get_result_async()
            rclpy.spin_until_future_complete(self, result, timeout_sec=120.0)
            if not result.done() or result.result().result.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(f'Home motion failed for {side} arm')
                return False
            self.get_logger().info(f'{side.capitalize()} Arm homed successfully!')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = DualArmHoming()
    try:
        success = node.go_home()
    finally:
        node.move_group.destroy()
        node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
