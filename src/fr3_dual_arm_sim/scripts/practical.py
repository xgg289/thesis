#!/usr/bin/env python3
"""Pick, stack and return a tray using the planner in rq3/4_obs1_fallback.py.

Requires a fresh practical_world.sdf and the workspace's MoveIt controllers.
Gazebo joints hold the grasped parts; the released block rests freely in the tray.
"""
import copy
import importlib.util
import math
import random
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, AllowedCollisionEntry,
                             Constraints, JointConstraint, OrientationConstraint)
from moveit_msgs.srv import GetPositionIK, GetPositionFK, ApplyPlanningScene
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_prefix

SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("practical_fallback", SCRIPTS / "rq3/4_obs1_fallback.py")
fallback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fallback)
msgs = Path(get_package_prefix("gz_msgs_vendor")) / "opt/gz_msgs_vendor/lib/python"
sys.path.append(str(msgs))
from gz.msgs10.scene_pb2 import Scene
from gz.msgs10.pose_v_pb2 import Pose_V
from google.protobuf.text_format import Parse, Merge

WORLD = "practical_world"
WORLD_FILE = SCRIPTS.parent / "worlds/practical_world.sdf"
DOWN_QUAT = (1.0, 0.0, 0.0, 0.0)

# Offsets between wrists and grasp points (m)
GRASP_OFFSETS = {"left": 0.1034, "right": 0.1134}
# Tray offsets
TRAY_RIM_Y = 0.096
TRAY_FAR_RIM_Y = -0.236
TRAY_RIM_Z = 0.0025
LEFT_GRIP_WIDTH = 0.0042  # 8.4 mm total opening

# Tray assembly position and block offset (m)
ASSEMBLY = (-0.30, 0.03, 1.14)
BLOCK_OFFSET = (0.0, -0.17, 0.008)  # relative to tray
TRAY_PLACED_Z = 0.9775  # on the pickup stand
TRAY_RELEASE_CLEARANCE = 0.001  # 1 mm above the stand

def quaternion_product(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z)

def rotate(q, point):
    conjugate = (-q[0], -q[1], -q[2], q[3])
    return quaternion_product(quaternion_product(q, (*point, 0.0)), conjugate)[:3]

def euler_quaternion(angles):
    roll, pitch, yaw = angles
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)

def stack_target(tray_centre):
    return tuple(a + b for a, b in zip(tray_centre, BLOCK_OFFSET))

def pose(position, orientation=(0.0, 0.0, 0.0, 1.0)):
    value = Pose()
    value.position.x, value.position.y, value.position.z = position
    value.orientation.x, value.orientation.y, value.orientation.z, value.orientation.w = orientation
    return value

class PlanClient:
    def __init__(self, client, node):
        self.client, self.node = client, node

    def __getattr__(self, name):
        return getattr(self.client, name)

    def call_async(self, request):
        if getattr(self.node, "combined_request", False) and self.node.fixed_joints:
            # Inactive arm fixed during combined planning.
            request.motion_plan_request.path_constraints = Constraints(
                joint_constraints=[JointConstraint(joint_name=name, position=value,
                    tolerance_above=0.001, tolerance_below=0.001, weight=1.0)
                    for name, value in self.node.fixed_joints.items()])
        plan = request.motion_plan_request
        if self.node.carry_tray and plan.group_name in ("left_fr3_arm", "both_arms"):
            # Tray level while it is carried.
            constraint = OrientationConstraint()
            constraint.header.frame_id = "world"
            constraint.link_name = "left_fr3_link8"
            # Hand mounting angle.
            rotation = quaternion_product(DOWN_QUAT,
                (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)))
            constraint.orientation = pose((0.0, 0.0, 0.0), rotation).orientation
            constraint.absolute_x_axis_tolerance = 0.1
            constraint.absolute_y_axis_tolerance = 0.1
            constraint.absolute_z_axis_tolerance = math.pi
            constraint.parameterization = OrientationConstraint.ROTATION_VECTOR
            constraint.weight = 1.0
            plan.path_constraints.orientation_constraints = [constraint]
        return self.client.call_async(request)

class GoalTimeToleranceClient:
    """Keep the practical demonstration's four-second goal tolerance."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def send_goal_async(self, goal, *args, **kwargs):
        goal.goal_time_tolerance = Duration(sec=4, nanosec=0)
        return self.client.send_goal_async(goal, *args, **kwargs)

class Practical(fallback.DualArmMoveGroupJointPose):
    def __init__(self):
        try:
            super().__init__()
            self._initialize_practical()
        except BaseException:
            try:
                self.destroy_node()
            except Exception:
                pass
            raise

    def _initialize_practical(self):
        self.exec_clients = [GoalTimeToleranceClient(c) for c in self.exec_clients]
        self.plan_client = PlanClient(self.plan_client, self)
        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.fk = self.create_client(GetPositionFK, "/compute_fk")
        self.grippers = {side: ActionClient(self, GripperCommand,
            f"/{side}_fr3_hand_controller/gripper_cmd") for side in ("left", "right")}
        self.move_results = []
        self.task_start = None
        self.current_stage = None
        self.grasp_plugins = set()
        self.carry_tray = False
        self.ik_start_state = None
        self.objects = {}
        self.sizes = {}
        for model in ET.parse(WORLD_FILE).getroot().findall("./world/model"):
            collisions = model.findall("./link/collision")
            if not collisions:
                continue
            shapes = []
            for collision in collisions:
                box = collision.find("./geometry/box/size")
                if box is None:
                    continue
                size = tuple(map(float, box.text.split()))
                offset = tuple(map(float, collision.findtext("pose", "0 0 0 0 0 0").split()[:3]))
                angles = tuple(map(float, collision.findtext("pose", "0 0 0 0 0 0").split()[3:]))
                rotation = euler_quaternion(angles)
                shapes.append((size, offset, rotation))
            if not shapes:
                continue
            position = tuple(map(float, model.findtext("pose").split()[:3]))
            name = model.get("name")
            self.sizes[name] = shapes
            self.objects[name] = self.box(name, position, shapes)

        # Tray and stand dimensions 
        walls = [offset for size, offset, _ in self.sizes["base_component"]
                 if abs(size[1] - 0.008) < 1e-6 and abs(size[2] - 0.024) < 1e-6]
        positive_rim = next((p[1] for p in walls if p[1] > 0), None)
        negative_rim = next((p[1] for p in walls if p[1] < 0), None)
        if "assembly_stand" in self.objects:
            raise RuntimeError("The middle assembly stand is still in the SDF. Use the return-to-pickup world.")
        stand = self.objects["left_pickup_stand"].pose.position
        tray = self.objects["base_component"].pose.position
        support = next(((size, offset) for size, offset, _ in self.sizes["left_pickup_stand"]
                        if size[0] >= 0.19 and size[1] >= 0.35 and size[2] <= 0.03), None)
        floor = next(((size, offset) for size, offset, _ in self.sizes["base_component"]
                      if size[0] >= 0.18 and size[1] >= 0.34 and size[2] <= 0.01), None)
        if (positive_rim is None or negative_rim is None or support is None or floor is None
                or abs(positive_rim - TRAY_RIM_Y) > 0.001
                or abs(negative_rim - TRAY_FAR_RIM_Y) > 0.001
                or abs(tray.z - TRAY_PLACED_Z) > 0.002):
            raise RuntimeError("Python and tray/stand SDF geometry do not match. Restart with the updated world.")
        deck_size, deck_offset = support
        floor_size, floor_offset = floor
        deck_center = (stand.x + deck_offset[0], stand.y + deck_offset[1])
        floor_center = (tray.x + floor_offset[0], tray.y + floor_offset[1])
        deck_top = stand.z + deck_offset[2] + deck_size[2] / 2
        floor_bottom = tray.z + floor_offset[2] - floor_size[2] / 2
        if (abs(deck_top - floor_bottom) > 0.003
                or abs(deck_center[0] - floor_center[0]) > 0.005
                or abs(deck_center[1] - floor_center[1]) > 0.005
                or deck_size[0] < floor_size[0] + 0.015
                or deck_size[1] < floor_size[1] + 0.015):
            raise RuntimeError("Left pickup stand must support the complete tray at the original pickup height.")
        if not (TRAY_FAR_RIM_Y + 0.008 + 0.0175 < BLOCK_OFFSET[1]
                < TRAY_RIM_Y - 0.008 - 0.0175):
            raise RuntimeError("The block placement is outside the tray's clear interior.")

    def configure_planner_sampling(self):
        super().configure_planner_sampling()
        # Joint-space sampling with tray orientation constraint.
        setter = self.create_client(fallback.SetPlannerParams, "/set_planner_params")
        getter = self.create_client(fallback.GetPlannerParams, "/get_planner_params")
        try:
            if not setter.wait_for_service(timeout_sec=10.0) or not getter.wait_for_service(timeout_sec=10.0):
                raise RuntimeError("Planner settings are unavailable.")
            request = fallback.SetPlannerParams.Request()
            request.pipeline_id, request.group = "ompl", "left_fr3_arm"
            request.planner_config = "RRTConnectkConfigDefault"
            request.params.keys = ["enforce_joint_model_state_space"]
            request.params.values = ["1"]
            self.wait(setter.call_async(request))
            check = fallback.GetPlannerParams.Request()
            check.pipeline_id, check.group, check.planner_config = request.pipeline_id, request.group, request.planner_config
            params = self.wait(getter.call_async(check)).params
            if dict(zip(params.keys, params.values)).get("enforce_joint_model_state_space") != "1":
                raise RuntimeError("Could not set joint-space sampling.")
        finally:
            self.destroy_client(setter)
            self.destroy_client(getter)

    def wait_for_joint_state(self, timeout_sec=30.0):
        return super().wait_for_joint_state(timeout_sec=timeout_sec)

    def box(self, name, centre, shapes):
        obj = CollisionObject()
        obj.header.frame_id, obj.id = "world", name
        obj.operation = CollisionObject.ADD
        obj.pose = pose(centre)
        obj.primitives = [fallback.SolidPrimitive(
            type=fallback.SolidPrimitive.BOX, dimensions=list(size)) for size, _, _ in shapes]
        obj.primitive_poses = [pose(offset, orientation) for _, offset, orientation in shapes]
        return obj

    def gz(self, args):
        result = subprocess.run(["gz", *args], capture_output=True, text=True, timeout=12)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "Gazebo command failed.")
        return result.stdout

    def gazebo_scene(self):
        output = self.gz(["service", "-s", f"/world/{WORLD}/scene/info",
            "--reqtype", "gz.msgs.Empty", "--reptype", "gz.msgs.Scene",
            "--timeout", "4000", "--req", ""])
        scene = Parse(output, Scene())
        return {model.name: model for model in scene.model}

    def component_pose(self, name):
        output = self.gz(["topic", "-e", "-t", f"/world/{WORLD}/pose/info", "-n", "1"])
        poses = Merge(output, Pose_V())
        value = next((p for p in reversed(poses.pose) if p.name == name), None)
        if value is None:
            model = self.gazebo_scene().get(name)
            if model is not None:
                value = next((p for p in reversed(poses.pose) if p.id == model.id), None)
        if value is None:
            raise RuntimeError(f"Live pose for {name} is unavailable. Launch {WORLD_FILE.name}.")
        return pose((value.position.x, value.position.y, value.position.z),
                    (value.orientation.x, value.orientation.y, value.orientation.z, value.orientation.w))

    def attachment_command(self, side, attached):
        if not attached and side not in self.grasp_plugins:
            return
        topic = f"/practical/{side}/state"
        desired = "attached" if attached else "detached"

        reader = subprocess.Popen(
            ["gz", "topic", "-e", "-t", topic],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            time.sleep(0.6)
            if attached and side not in self.grasp_plugins:
                model = self.gazebo_scene()["dual_fr3"]
                part = "base_component" if side == "left" else "top_component"
                xml = (f"<parent_link>{side}_fr3_link7</parent_link>"
                       f"<child_model>{part}</child_model><child_link>body</child_link>"
                       f"<attach_topic>/practical/{side}/attach</attach_topic>"
                       f"<detach_topic>/practical/{side}/detach</detach_topic>"
                       f"<output_topic>{topic}</output_topic>")
                payload = (f'entity {{ id: {model.id} name: "dual_fr3" type: MODEL }} '
                           'plugins { filename: "gz-sim-detachable-joint-system" '
                           'name: "gz::sim::systems::DetachableJoint" innerxml: "'+xml+'" }')
                response = self.gz(["service", "-s", f"/world/{WORLD}/entity/system/add",
                    "--reqtype", "gz.msgs.EntityPlugin_V", "--reptype", "gz.msgs.Boolean",
                    "--timeout", "4000", "--req", payload])
                if "data: true" not in response:
                    raise RuntimeError(f"Could not create the {side} grasp joint.")
                self.grasp_plugins.add(side)
            else:
                self.gz(["topic", "-t", f"/practical/{side}/{'attach' if attached else 'detach'}",
                         "-m", "gz.msgs.Empty", "-p", " "])

            # Wait for Gazebo to confirm attachment state.
            import selectors
            selector = selectors.DefaultSelector()
            selector.register(reader.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 12.0
            output = []
            while time.monotonic() < deadline:
                events = selector.select(timeout=min(0.5, max(0, deadline - time.monotonic())))
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    output.append(line)
                    if f'"{desired}"' in line:
                        return
                if reader.poll() is not None:
                    break
            raise RuntimeError(
                f"Gazebo did not confirm {side} {desired} on {topic} within 12 s. "
                f"Received: {''.join(output).strip() or '(no messages)'}. "
                f"Inspect with: gz topic -i -t {topic}"
            )
        finally:
            if reader.poll() is None:
                reader.kill()
            reader.communicate()

    def apply(self, world=(), attached=(), matrix=None):
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = request.scene.robot_state.is_diff = True
        request.scene.world.collision_objects = list(world)
        request.scene.robot_state.attached_collision_objects = list(attached)
        if matrix is not None:
            request.scene.allowed_collision_matrix = matrix
        if not self.wait(self.apply_client.call_async(request)).success:
            raise RuntimeError("MoveIt rejected the component scene update.")

    def allow(self, name, links, enabled):
        acm = self.get_scene().allowed_collision_matrix
        for entry in (name, *links):
            if entry not in acm.entry_names:
                acm.entry_names.append(entry)
                for row in acm.entry_values:
                    row.enabled.append(False)
                acm.entry_values.append(AllowedCollisionEntry(enabled=[False] * len(acm.entry_names)))
        i = acm.entry_names.index(name)
        for link in links:
            j = acm.entry_names.index(link)
            acm.entry_values[i].enabled[j] = acm.entry_values[j].enabled[i] = enabled
        self.apply(matrix=acm)

    def fingers(self, side, position):
        goal = GripperCommand.Goal()
        goal.command.position, goal.command.max_effort = float(position), 20.0
        future = self.grippers[side].send_goal_async(goal)
        handle = self.wait(future)
        if handle is None or not handle.accepted:
            raise RuntimeError(f"{side} gripper rejected its command.")
        result_future = handle.get_result_async()
        try:
            response = self.wait(result_future, 15.0)
            if response.status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f"{side} gripper failed: {response}")
            # Gripper can stall against an object without reaching commanded width.
            outcome = response.result
            self.get_logger().info(
                f"{side} gripper: reached_goal={getattr(outcome, 'reached_goal', 'N/A')}, "
                f"stalled={getattr(outcome, 'stalled', 'N/A')}")
        except BaseException:
            handle.cancel_goal_async()
            raise

    def build_start_state(self):
        state = super().build_start_state()
        # Include carried objects in planning start state.
        state.attached_collision_objects = copy.deepcopy(
            self.get_scene().robot_state.attached_collision_objects)
        return state

    def link_pose(self, side):
        req = GetPositionFK.Request()
        req.header.frame_id = "world"
        req.fk_link_names = [f"{side}_fr3_link8"]
        req.robot_state = self.build_start_state()
        result = self.wait(self.fk.call_async(req))
        if result.error_code.val != 1:
            raise RuntimeError("Forward kinematics failed.")
        return result.pose_stamped[0].pose

    def attach_scene(self, side, name):
        # Wait for joints to settle before reading grasp transform.
        if not self.wait_for_joint_state(timeout_sec=20.0):
            raise RuntimeError(
                f"{side} arm joints did not settle after grasp. "
                "Check /joint_states and gripper/tray contact in Gazebo."
            )
        world = self.component_pose(name)
        parent = self.link_pose(side)
        q = (parent.orientation.x, parent.orientation.y, parent.orientation.z, parent.orientation.w)
        inverse = (-q[0], -q[1], -q[2], q[3])
        delta = (world.position.x-parent.position.x, world.position.y-parent.position.y,
                 world.position.z-parent.position.z)
        orientation = (world.orientation.x, world.orientation.y, world.orientation.z, world.orientation.w)
        obj = self.box(name, (0.0, 0.0, 0.0), self.sizes[name])
        obj.header.frame_id = f"{side}_fr3_link8"
        obj.pose = pose(rotate(inverse, delta), quaternion_product(inverse, orientation))
        attached = AttachedCollisionObject()
        attached.link_name, attached.object = f"{side}_fr3_link8", obj
        attached.touch_links = [f"{side}_fr3_link8", f"{side}_fr3_hand",
                               f"{side}_fr3_leftfinger", f"{side}_fr3_rightfinger"]
        self.apply(attached=[attached])
        scene = self.get_scene()
        if name not in {a.object.id for a in scene.robot_state.attached_collision_objects}:
            raise RuntimeError(f"MoveIt did not attach {name}.")

    def detach_scene(self, side, name):
        attached = AttachedCollisionObject()
        attached.link_name = f"{side}_fr3_link8"
        attached.object.id, attached.object.operation = name, CollisionObject.REMOVE
        obj = copy.deepcopy(self.objects[name])
        obj.pose = self.component_pose(name)
        self.apply(world=[obj], attached=[attached])

    def joint_goal(self, side, centre, orientation):
        state = copy.deepcopy(self.ik_start_state) if self.ik_start_state is not None else self.build_start_state()
        orientation = quaternion_product(orientation,
            (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)))
        # Grasp point into a wrist target.
        offset = rotate(orientation, (0.0, 0.0, GRASP_OFFSETS[side]))
        target = list(centre[i]-offset[i] for i in range(3))
        names = self.left_home if side == "left" else self.right_home
        if not hasattr(self, "position_limits"):
            response = self.wait(self.limit_client.call_async(
                fallback.GetParameters.Request(names=["robot_description"])))
            robot = ET.fromstring(response.values[0].string_value)
            self.position_limits = {j.get("name"): (float(j.find("limit").get("lower")),
                float(j.find("limit").get("upper"))) for j in robot.findall("joint")
                if j.find("limit") is not None and j.find("limit").get("lower") is not None}
        generator = random.Random(21)
        for attempt in range(32):
            request = GetPositionIK.Request()
            request.ik_request.group_name = f"{side}_fr3_arm"
            request.ik_request.ik_link_name = f"{side}_fr3_link8"
            request.ik_request.robot_state = copy.deepcopy(state)
            if attempt:
                for i, name in enumerate(state.joint_state.name):
                    if name in names:
                        low, high = self.position_limits[name]
                        current = state.joint_state.position[i]
                        value = current + generator.uniform(-0.5, 0.5) if attempt < 16 else generator.uniform(low, high)
                        request.ik_request.robot_state.joint_state.position[i] = min(high-0.01, max(low+0.01, value))
            request.ik_request.avoid_collisions = True
            request.ik_request.timeout = Duration(sec=1)
            request.ik_request.pose_stamped.header.frame_id = "world"
            request.ik_request.pose_stamped.pose = pose(target, orientation)
            response = self.wait(self.ik.call_async(request), 5.0)
            if response.error_code.val == 1:
                joints = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
                return {name: joints[name] for name in names}
        raise RuntimeError(f"No collision-free IK for {side} at {centre}.")

    def hold_goal(self, side):
        state = self.build_start_state()
        joints = dict(zip(state.joint_state.name, state.joint_state.position))
        return {name: joints[name] for name in (self.left_home if side == "left" else self.right_home)}

    def move(self, label, left=None, right=None):
        self.current_stage = label
        self.get_logger().info(label)
        status = "START STATE FAILED"
        try:
            self.ik_start_state = self.build_start_state()
            status = "LEFT IK FAILED" if left else "LEFT HOLD STATE FAILED"
            left_goal = self.joint_goal("left", *left) if left else self.hold_goal("left")
            # Right-arm IK with left arm at its target.
            joints = self.ik_start_state.joint_state
            joints.position = [
                left_goal.get(name, value)
                for name, value in zip(joints.name, joints.position)
            ]
            status = "RIGHT IK FAILED" if right else "RIGHT HOLD STATE FAILED"
            right_goal = self.joint_goal("right", *right) if right else self.hold_goal("right")
        except Exception as error:
            reason = str(error) or type(error).__name__
            result = dict(planning=None, checking=None, execution=None, length=None,
                          targets=0, success=False, status=status, attempts=0,
                          planning_failures=0, checked_pairs=0, passed_pairs=0,
                          valid_plans=0, fallback_needed=False, fallback_reason=None,
                          strategy="NONE", failure_reason=reason, collision_details=[])
            self.move_results.append((label, left is not None, right is not None, result))
            self.get_logger().error(f"{label}: {status}: {reason}")
            raise RuntimeError(f"{label}: {status}: {reason}") from error
        finally:
            self.ik_start_state = None
        self.fixed_joints = left_goal if left is None else right_goal if right is None else None
        try:
            result = self.move_pair(left_goal, right_goal)
        finally:
            self.fixed_joints = None
        self.move_results.append((label, left is not None, right is not None, result.copy()))
        if not result["success"]:
            raise RuntimeError(f"{label} failed: {result['status']}")

    def print_trial_results(self, elapsed, finished, error=None):
        # Available timing measurements from failed movements.
        moves = self.move_results
        results = [item[3] for item in moves]

        def duration(key, only_active=None):
            values = [r.get(key) for _, left, right, r in moves
                      if (only_active is None
                          or (only_active == "left" and left)
                          or (only_active == "right" and right))
                      and r.get(key) is not None]
            return "N/A" if not values else f"{sum(values):.4f} s"

        def count(key):
            return sum(int(r.get(key) or 0) for r in results)

        lengths = [r["length"] for r in results
                   if r.get("success") and r.get("length") is not None]
        checked = count("checked_pairs")
        passed = count("passed_pairs")
        independent = sum(r.get("success") and r.get("strategy") == "independent"
                          for r in results)
        combined = sum(r.get("success") and r.get("strategy") == "combined"
                       for r in results)
        # Record selected plan even if execution fails.
        selected_strategies = {
            r.get("strategy") for r in results
            if (r.get("valid_plans") or r.get("success"))
            and r.get("strategy") in ("independent", "combined")
        }
        selected = ("MIXED" if len(selected_strategies) == 2 else
                    "INDEPENDENT" if "independent" in selected_strategies else
                    "COMBINED" if "combined" in selected_strategies else "NONE")
        status = ("FAILED" if error is not None else
                  "SUCCESS" if finished else "INCOMPLETE")

        rows = [
            ("Planning time", duration("planning")),
            ("Collision-checking time", duration("checking")),
            ("Trajectory length (14D)",
             "N/A" if not lengths else f"{sum(lengths):.4f} rad"),
            ("Execution time", duration("execution")),
            ("Task completion time" if status == "SUCCESS" else "Elapsed task time",
             "N/A" if elapsed is None else f"{elapsed:.4f} s"),
            ("Successful movements", str(sum(bool(r.get("success")) for r in results))),
            ("Pair collision check", "NOT COMPLETED" if not checked else
             "PASS" if checked == passed else "MIXED/FAILED"),
            ("Collision details", "; ".join(sorted({detail for r in results
                for detail in r.get("collision_details", [])})) or "N/A"),
            ("Planning attempts", str(count("attempts"))),
            ("Planning failures", str(count("planning_failures"))),
            ("Collision-free pairs", f"{passed}/{checked}"),
            ("Selected plan", selected),
            ("Independent movements", str(independent)),
            ("Combined movements", str(combined)),
            ("Fallback used", f"{sum(bool(r.get('fallback_needed')) for r in results)} time(s)"),
            ("Task success", status),
        ]
        lines = ["", "================ TRIAL RESULTS ================"]
        lines.extend(f"{label:<29} {value:>16}" for label, value in rows)
        if error is not None:
            lines.append(f"Failed stage: {self.current_stage or 'Setup/verification'}")
            lines.append(f"Status: {error}")
        else:
            lines.append(f"Status: {status}")
        lines.append("================================================")
        self.get_logger().info("\n".join(lines))

    def plan_combined(self, group_name, targets, start_state):
        self.combined_request = True
        try:
            return super().plan_combined(group_name, targets, start_state)
        finally:
            self.combined_request = False

    def verify_stack(self, stage, tolerance=0.015):
        """Check the block position and tray tilt."""
        self.current_stage = stage
        tray = self.component_pose("base_component")
        block = self.component_pose("top_component")
        q = tray.orientation
        orientation = (q.x, q.y, q.z, q.w)
        inverse = (-q.x, -q.y, -q.z, q.w)
        delta = (block.position.x - tray.position.x,
                 block.position.y - tray.position.y,
                 block.position.z - tray.position.z)
        relative = rotate(inverse, delta)
        deviation = math.dist(relative, BLOCK_OFFSET)
        tilt = rotate(orientation, (0.0, 0.0, 1.0))[2]
        if tilt < math.cos(0.15):
            raise RuntimeError(f"{stage}: tray has tilted more than 0.15 rad.")
        if deviation > tolerance:
            raise RuntimeError(
                f"{stage}: loose block shifted {deviation*1000:.1f} mm. "
                "Stop rather than continuing with an inaccurate planning scene.")

    def validate_grasp(self, side, intended):
        # Check both grasp positions before attaching objects.
        wrist = self.link_pose(side)
        q = (wrist.orientation.x, wrist.orientation.y,
             wrist.orientation.z, wrist.orientation.w)
        tcp_offset = rotate(q, (0.0, 0.0, GRASP_OFFSETS[side]))
        actual = tuple(a + b for a, b in zip(
            (wrist.position.x, wrist.position.y, wrist.position.z), tcp_offset))
        error = math.dist(actual, intended)
        if error > 0.01:
            raise RuntimeError(
                f"{side.capitalize()} grasp position is {error * 1000:.1f} mm "
                "from its target (limit: 10 mm).")

    def run(self):
        topics = self.gz(["topic", "-l"]).splitlines()
        if any(f"/practical/{side}/state" in topics for side in ("left", "right")):
            raise RuntimeError("Restart the practical world before starting another demonstration.")
        # Verify that Gazebo is running practical world.
        live_models = self.gazebo_scene()
        if ("assembly_stand" in live_models
                or "left_pickup_stand" not in live_models
                or "base_component" not in live_models):
            raise RuntimeError("Gazebo is not using the return-to-pickup world. Restart with the new SDF.")
        self.wait_for_servers()
        for client in (self.ik, self.fk):
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError("MoveIt kinematics service unavailable.")
        for client in self.grippers.values():
            if not client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError("Gripper controller unavailable.")
        self.load_joint_limits()
        self.set_motion_scaling(1.0, 1.0)
        self.configure_planner_sampling()
        self.enable_cross_arm_checks()
        self.fixed_joints = None
        for side in ("left", "right"):
            self.attachment_command(side, False)
        for name in ("base_component", "top_component"):
            obj = self.objects[name]
            p = obj.pose.position
            payload = (
                f'name: "{name}" position {{x:{p.x} y:{p.y} z:{p.z}}} orientation {{w:1}}'
            )
            response = self.gz([
                "service", "-s", f"/world/{WORLD}/set_pose",
                "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                "--timeout", "4000", "--req", payload,
            ])
            if "data: true" not in response:
                raise RuntimeError("Component reset failed.")
        time.sleep(0.5)
        initial = self.build_start_state()
        initial_joints = dict(zip(initial.joint_state.name, initial.joint_state.position))
        for name in ("base_component", "top_component"):
            self.objects[name].pose = self.component_pose(name)
        self.apply(world=self.objects.values())
        for name in self.objects:
            self.allow(name, ["base"], name.endswith("_stand"))
        # Allow contact with pickup stands until parts are lifted.
        self.allow("base_component", ["left_pickup_stand"], True)
        self.allow("top_component", ["right_pickup_stand"], True)
        for side, name in (("left", "base_component"), ("right", "top_component")):
            self.allow(name, [f"{side}_fr3_hand", f"{side}_fr3_leftfinger",
                              f"{side}_fr3_rightfinger"], True)
            stand = "left_pickup_stand" if side == "left" else "right_pickup_stand"
            self.allow(stand, [f"{side}_fr3_hand", f"{side}_fr3_leftfinger",
                               f"{side}_fr3_rightfinger"], True)
            self.fingers(side, 0.04)

        def centre(name):
            p = self.objects[name].pose.position
            return (p.x, p.y, p.z)

        left_pick, right_pick = centre("base_component"), centre("top_component")
        tray_home = left_pick
        # Left gripper approaches +Y rim.
        left_pick = (left_pick[0], left_pick[1] + TRAY_RIM_Y, left_pick[2] + TRAY_RIM_Z)
        left_above = (left_pick[0], left_pick[1], left_pick[2] + 0.12)
        right_above = (right_pick[0], right_pick[1], right_pick[2] + 0.13)
        self.task_start = time.perf_counter()
        self.move("Approach both components", (left_above, DOWN_QUAT),
                  (right_above, DOWN_QUAT))
        self.move("Move into grasp positions", (left_pick, DOWN_QUAT),
                  (right_pick, DOWN_QUAT))
        if not self.wait_for_joint_state(timeout_sec=20.0):
            raise RuntimeError(
                "Arms did not settle at the grasp pose. "
                "Inspect /joint_states and possible tray/finger contact.")
        self.validate_grasp("left", left_pick)
        self.validate_grasp("right", right_pick)
        for side, name, width in (("left", "base_component", LEFT_GRIP_WIDTH),
                                  ("right", "top_component", 0.019)):
            self.fingers(side, width)
            self.attachment_command(side, True)
            self.attach_scene(side, name)
        self.carry_tray = True
        assembly = ASSEMBLY
        stack = stack_target(assembly)
        above = (stack[0] - 0.11, stack[1], stack[2] + 0.065)
        self.move("Bring components to shared workspace",
                  ((assembly[0], assembly[1] + TRAY_RIM_Y,
                    assembly[2] + TRAY_RIM_Z), DOWN_QUAT),
                  (above, DOWN_QUAT))
        # Remove pickup contact allowances after lifting.
        self.allow("base_component", ["left_pickup_stand"], False)
        self.allow("top_component", ["right_pickup_stand"], False)
        self.allow("left_pickup_stand", ["left_fr3_hand", "left_fr3_leftfinger",
                                        "left_fr3_rightfinger"], False)
        self.allow("right_pickup_stand", ["right_fr3_hand", "right_fr3_leftfinger",
                                         "right_fr3_rightfinger"], False)
        self.allow("base_component", ["top_component"], True)
        self.move("Left holds base; right places component on top", right=(stack, DOWN_QUAT))
        self.attachment_command("right", False)
        self.fingers("right", 0.04)
        self.detach_scene("right", "top_component")
        time.sleep(0.5)
        self.verify_stack("After right gripper release")
        # Block stays loose in Gazebo but follows tray in MoveIt.
        self.attach_scene("left", "top_component")
        if not self.wait_for_joint_state(timeout_sec=20.0):
            raise RuntimeError("Arms did not settle after the block was released.")
        self.move("Coordinated withdrawal",
                  left=((assembly[0] - 0.06, assembly[1] + TRAY_RIM_Y + 0.06,
                         assembly[2] + TRAY_RIM_Z), DOWN_QUAT),
                  right=((stack[0], stack[1] - 0.16, stack[2] + 0.10), DOWN_QUAT))
        self.verify_stack("After right arm withdrawal")

        # Lower tray to 1 mm above its stand before release.
        self.move("Return and lower tray onto pickup stand",
                  left=((tray_home[0], tray_home[1] + TRAY_RIM_Y,
                         TRAY_PLACED_Z + TRAY_RELEASE_CLEARANCE + TRAY_RIM_Z), DOWN_QUAT))
        self.verify_stack("Before release at original pickup stand")

        self.attachment_command("left", False)
        self.carry_tray = False
        self.fingers("left", 0.04)
        self.detach_scene("left", "top_component")
        self.detach_scene("left", "base_component")
        self.allow("base_component", ["left_pickup_stand"], True)
        time.sleep(1.2)
        self.verify_stack("After setting tray on pickup stand", tolerance=0.02)

        self.move("Left retreats above original pickup position",
                  left=(left_above, DOWN_QUAT))
        for name in ("base_component", "top_component"):
            for side in ("left", "right"):
                self.allow(name, [f"{side}_fr3_hand", f"{side}_fr3_leftfinger",
                                  f"{side}_fr3_rightfinger"], False)

        self.get_logger().info("Right returns HOME; left remains at the pickup station.")
        self.current_stage = "Right returns HOME; left stays at pickup"
        left_hold = self.hold_goal("left")
        self.fixed_joints = left_hold
        try:
            result = self.move_pair(left_hold,
                                    {n: initial_joints[n] for n in self.right_home})
        finally:
            self.fixed_joints = None
        self.move_results.append((self.current_stage, False, True, result.copy()))
        if not result["success"]:
            raise RuntimeError("Right arm failed to return HOME.")

        self.current_stage = "Final placement verification"
        placed = {}
        expected_positions = {
            "base_component": (tray_home[0], tray_home[1], TRAY_PLACED_Z),
            "top_component": (tray_home[0] + BLOCK_OFFSET[0],
                              tray_home[1] + BLOCK_OFFSET[1],
                              TRAY_PLACED_Z + BLOCK_OFFSET[2]),
        }
        for name, expected in expected_positions.items():
            placed[name] = self.component_pose(name)
            actual = placed[name].position
            if math.dist((actual.x, actual.y, actual.z), expected) > 0.02:
                raise RuntimeError(f"Final placement check failed for {name}.")
        tray, block = placed["base_component"], placed["top_component"]
        q = tray.orientation
        inverse = (-q.x, -q.y, -q.z, q.w)
        separation = rotate(inverse, (
            block.position.x - tray.position.x,
            block.position.y - tray.position.y,
            block.position.z - tray.position.z,
        ))
        if math.dist(separation, BLOCK_OFFSET) > 0.015:
            raise RuntimeError("The block shifted inside the tray during placement.")
        if rotate((q.x, q.y, q.z, q.w), (0.0, 0.0, 1.0))[2] < math.cos(0.1):
            raise RuntimeError("The placed tray is tilted.")
        for side in ("left", "right"):
            self.fingers(side, initial_joints.get(f'{side}_fr3_finger_joint1', 0.0))
        self.get_logger().info("DEMO SUCCESS: tray and block on original left pickup stand; "
                               "left gripper above pickup position; right arm returned HOME.")

def main():
    node = None
    finished = False
    error_message = None
    try:
        rclpy.init(args=["--ros-args", "-r", "__node:=practical_demo", *sys.argv[1:]])
        node = Practical()
        node.run()
        finished = True
    except BaseException as error:
        error_message = str(error) or type(error).__name__
        if node is None:
            rclpy.logging.get_logger("practical").error(
                f"Failed stage: Initialization. Failure reason: {error_message}")
        raise
    finally:
        try:
            if node is not None:
                elapsed = (time.perf_counter() - node.task_start
                           if node.task_start is not None else None)
                node.print_trial_results(elapsed, finished, error_message)
        finally:
            try:
                if node is not None:
                    node.destroy_node()
            finally:
                rclpy.try_shutdown()

if __name__ == "__main__":
    main()