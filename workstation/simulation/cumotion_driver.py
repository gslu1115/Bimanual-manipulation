"""Isaac 6.1 controller adapter: calibrated priors, encoder FK and visual objects only."""
from __future__ import annotations

import math
import time
import numpy as np

from workstation.observations.camera_geometry import quaternion_from_matrix
from workstation.observations.observation_packet import PANDA_ARM_JOINT_NAMES, PANDA_FINGER_JOINT_NAMES
from workstation.planning.collision import scene_obstacles


def merge_named_command(current, names, values, joint_names):
    """Copy SDK's mutable output immediately, resolving joints by name."""
    result = np.asarray(current, float).copy()
    values = np.asarray(values, float).reshape(-1)
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError('Invalid named cuMotion command')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite cuMotion command')
    for name, value in zip(names, values):
        if name not in joint_names:
            raise ValueError('Unknown cuMotion joint: '+name)
        result[joint_names.index(name)] = value
    return result


class CuMotionPickPlaceDriver:
    """Composable official PickPlaceController; no SceneQuery or USD object reads."""
    def __init__(self, robot, arm, scene, target_id, pick, pick_rotation, place, place_rotation,
                 close_position, approach_height=.18):
        import warp as wp
        import isaacsim.robot_motion.experimental.motion_generation as mg
        from isaacsim.robot_motion.cumotion import CumotionWorldInterface, RmpFlowController, load_cumotion_supported_robot
        from isaacsim.robot_motion.examples.manipulation import PickPlaceController, JointGripperController, GripperCommand

        self.wp, self.mg = wp, mg
        self.robot, self.arm, self.target_id = robot, arm, target_id
        self.names = list(PANDA_ARM_JOINT_NAMES+PANDA_FINGER_JOINT_NAMES)
        self.tool = 'panda_hand'
        self.time_s = 0.
        self.controller_wall_s = 0.
        self.calls = 0
        self.world = CumotionWorldInterface(device='cpu')
        base = robot.base_config[arm]
        yaw = math.radians(base['yaw_deg'])/2
        self.world.update_world_to_robot_root_transforms((self.array([base['base_xyz']]),
            self.array([[math.cos(yaw), 0., 0., math.sin(yaw)]])))
        # The base's mounting contact is handled by the project's link-specific
        # guard. Excluding this obstacle from RMP attraction avoids base repulsion.
        self.add_boxes([b for b in robot.static if b.name != arm+'_mount'], 'fixed')
        self.scene_paths = []
        self.scene_revision = 0
        self.update_scene(scene)
        states = robot.read_states()
        other = next(name for name in states if name != arm)
        self.other_arm = other
        spheres = robot.spheres(other, states[other].joint_positions_rad,
                                float(max(states[other].finger_positions_m)))
        self.other_q = states[other].qpos.copy()
        self.world.add_spheres([f'/Observed/OtherArm/{i}' for i in range(len(spheres))],
            self.array([s[2] for s in spheres]), self.array(np.ones((len(spheres), 3))),
            self.array(np.full(len(spheres), .002)),
            (self.array([s[1] for s in spheres]), self.array(np.tile([1., 0., 0., 0.], (len(spheres), 1)))),
            wp.array(np.ones(len(spheres), bool), dtype=wp.bool, device='cpu'))
        # Measure the fixed hand->existing project TCP from robot-model FK.
        q = states[arm].joint_positions_rad
        hand, hand_r = robot.fk(arm, q, self.tool)
        tcp, tcp_r = robot.fk(arm, q)
        self.hand_to_tcp = np.asarray(hand_r).T@(np.asarray(tcp)-hand)
        self.hand_to_tcp_r = np.asarray(hand_r).T@tcp_r
        native = load_cumotion_supported_robot('franka')
        self.arm_controller = RmpFlowController(native, self.world, self.names, [self.tool], tool_frame=self.tool)
        gripper_args = dict(robot_joint_space=self.names, joint_names=list(PANDA_FINGER_JOINT_NAMES),
            open_positions=[.04, .04], closed_positions=[close_position]*2,
            position_tolerance=.002, velocity_tolerance=.1, minimum_close_fraction=.1)
        self.controller = PickPlaceController(arm_controller=self.arm_controller,
            gripper_open_controller=JointGripperController(**gripper_args, command=GripperCommand.OPEN),
            gripper_close_controller=JointGripperController(**gripper_args, command=GripperCommand.CLOSE),
            robot_site_space=[self.tool], tool_frame=self.tool,
            controller_to_grasp_position=tuple(self.hand_to_tcp),
            controller_to_grasp_orientation=tuple(quaternion_from_matrix(self.hand_to_tcp_r)),
            approach_height=approach_height, lift_height=.18,
            position_tolerance=.003, grasp_position_tolerance=.003, orientation_tolerance=.05)
        self.goal = self.goal_state(pick, pick_rotation, place, place_rotation)
        if not self.controller.reset(self.read_state(), self.goal, 0.):
            raise RuntimeError('CUMOTION_RESET:'+str(self.controller.failure_reason))

    def array(self, values):
        return self.wp.array(np.asarray(values, np.float32), dtype=self.wp.float32, device='cpu')

    def add_boxes(self, boxes, group):
        if not boxes:
            return []
        paths = [f'/Observed/{group}/{i}' for i in range(len(boxes))]
        count = len(boxes)
        self.world.add_cubes(paths, self.array(np.ones(count)), self.array([b.size for b in boxes]),
            self.array(np.full(count, .002)),
            (self.array([b.centre for b in boxes]), self.array(np.tile([1., 0., 0., 0.], (count, 1)))),
            self.wp.array(np.ones(count, bool), dtype=self.wp.bool, device='cpu'))
        return paths

    def update_scene(self, scene):
        if self.scene_paths:
            self.world.update_obstacle_enables(self.scene_paths,
                self.wp.array(np.zeros(len(self.scene_paths), bool), dtype=self.wp.bool, device='cpu'))
        self.scene_revision += 1
        self.scene_paths = self.add_boxes(scene_obstacles(scene, self.target_id), f'Visual{self.scene_revision}')

    def goal_state(self, pick, pick_rotation, place, place_rotation):
        names = ['pick', 'place']
        return self.mg.RobotState(sites=self.mg.SpatialState.from_name(spatial_space=names,
            positions=(names, self.array([pick, place])),
            orientations=(names, self.array([quaternion_from_matrix(pick_rotation), quaternion_from_matrix(place_rotation)]))))

    def read_state(self):
        states = self.robot.read_states()
        if np.max(np.abs(states[self.other_arm].qpos-self.other_q)) > .01:
            raise RuntimeError('CUMOTION_STATIONARY_OTHER_ARM_MOVED')
        measured = states[self.arm]
        if measured.qvel is None:
            raise RuntimeError('CUMOTION_REQUIRES_MEASURED_JOINT_VELOCITY')
        p, r = self.robot.fk(self.arm, measured.joint_positions_rad, self.tool)
        return self.mg.RobotState(joints=self.mg.JointState.from_name(robot_joint_space=self.names,
            positions=(self.names, self.array(measured.qpos)), velocities=(self.names, self.array(measured.qvel))),
            sites=self.mg.SpatialState.from_name(spatial_space=[self.tool],
                positions=([self.tool], self.array([p])),
                orientations=([self.tool], self.array([quaternion_from_matrix(r)]))))

    @property
    def phase(self):
        return self.controller.phase.name

    @property
    def done(self):
        return self.controller.is_done

    def update_place_from_measurement(self, position, rotation):
        # Isaac 6.1 snapshots goals at reset and exposes no public retarget API.
        # This narrowly pinned integration updates only its validated place goal
        # at an observed phase boundary, never rewinds phases or opens fingers.
        if self.phase not in ('LIFT', 'APPROACH_PLACE', 'DESCEND_PLACE'):
            raise RuntimeError('CUMOTION_PLACE_UPDATE_OUTSIDE_OBSERVED_BOUNDARY')
        goal = self.goal_state(position, rotation, position, rotation)
        self.controller._place = self.controller._read_goal(goal, self.controller.PLACE_SITE)

    def propose(self, duration):
        self.time_s += duration
        started = time.perf_counter()
        desired = self.controller.forward(self.read_state(), None, self.time_s)
        self.controller_wall_s += time.perf_counter()-started
        self.calls += 1
        if self.controller.failed or desired is None:
            raise RuntimeError('CUMOTION:'+str(self.controller.failure_reason))
        target = self.robot.targets[self.arm].copy()
        velocity = np.zeros(9)
        if desired.joints is not None:
            joints = desired.joints
            if joints.positions is not None:
                target = merge_named_command(target, joints.position_names, joints.positions.numpy(), self.names)
            if joints.velocities is not None:
                velocity = merge_named_command(velocity, joints.velocity_names, joints.velocities.numpy(), self.names)
        return target, velocity
