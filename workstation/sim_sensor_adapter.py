"""Whitelist the three physical-camera-equivalent inputs from Isaac Sim.

Instantiate only after SimulationApp and SortingEnvironment.start(). The existing
CameraSystem.observe() remains a diagnostic API and may contain privileged data.
This adapter creates a new packet from selected RGB-D and measured robot DOFs;
renderer object labels and renderer camera world poses never leave this module.
"""
from __future__ import annotations

import math

import numpy as np

from camera_config import FORMAL_CAMERAS
from camera_geometry import USD_FROM_OPENCV, look_at_quaternion, pose_matrix
from observation_packet import (FORMAL_CAMERA_NAMES, ROBOT_NAMES, CameraFrame,
                                ObservationPacket, RobotState)


WRIST_ROBOTS = {
    'left_wrist_camera': 'panda_left',
    'right_wrist_camera': 'panda_right',
}


class SimSensorAdapter:
    """Build a deployable observation packet using nominal simulated calibration.

    The scene pose and hand-to-camera mounts come from the configured workcell;
    a RealSensorAdapter must replace them with measured calibration. Wrist world
    poses are computed from measured joint positions and Lula hand FK.
    """

    def __init__(self, env, max_frame_age_s=0.15):
        if tuple(FORMAL_CAMERAS) != FORMAL_CAMERA_NAMES:
            raise RuntimeError('Formal camera names disagree with the packet contract')
        if not math.isfinite(max_frame_age_s) or max_frame_age_s <= 0:
            raise ValueError('max_frame_age_s must be finite and positive')
        self.env = env
        self.max_frame_age_s = float(max_frame_age_s)
        self.last_errors = {}
        cameras = {c['name']: c for c in env.c['cameras'] if c['name'] in FORMAL_CAMERA_NAMES}
        if set(cameras) != set(FORMAL_CAMERA_NAMES):
            raise ValueError('Configuration must define the three formal cameras')
        self._camera_config = cameras
        self._T_workcell_from_scene_cv = self._scene_pose(cameras['scene_camera'])
        self._T_hand_from_camera_cv = {
            name: pose_matrix(cameras[name]['mount_translation'],
                              cameras[name]['mount_orientation_wxyz']) @ USD_FROM_OPENCV
            for name in WRIST_ROBOTS
        }

        # Import Isaac only after SimulationApp exists. The solver's panda_hand
        # frame matches the link under which each wrist camera is authored.
        from isaacsim.robot_motion.motion_generation import (LulaKinematicsSolver,
            load_supported_lula_kinematics_solver_config)
        robot_config = {r['name']: r for r in env.c['robots']}
        if set(robot_config) != set(ROBOT_NAMES) or set(env.arms) != set(ROBOT_NAMES):
            raise ValueError('Expected exactly the two configured Panda arms')
        self._solvers = {}
        for name in ROBOT_NAMES:
            solver = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config('Franka'))
            base = robot_config[name]
            half_yaw = math.radians(base['yaw_deg']) / 2
            solver.set_robot_base_pose(np.asarray(base['base_xyz'], dtype=float),
                                       np.array([math.cos(half_yaw), 0., 0., math.sin(half_yaw)]))
            if 'panda_hand' not in solver.get_all_frame_names():
                raise RuntimeError(f'Lula has no panda_hand frame for {name}')
            if env.arms[name].dof_names[:7] != solver.get_joint_names():
                raise RuntimeError(f'Panda joint order differs from Lula for {name}')
            self._solvers[name] = solver

    @staticmethod
    def _scene_pose(config):
        if 'orientation_wxyz' in config:
            q = config['orientation_wxyz']
        else:
            q = look_at_quaternion(np.asarray(config['position'], dtype=float),
                                   np.asarray(config['look_at'], dtype=float),
                                   np.asarray(config['up'], dtype=float))
        return pose_matrix(config['position'], q) @ USD_FROM_OPENCV

    @staticmethod
    def _robot_states(joint_positions, sample_time_s):
        if set(joint_positions) != set(ROBOT_NAMES):
            raise ValueError('Joint snapshot must contain both Panda arms')
        return {name: RobotState(sample_time_s=sample_time_s,
                                 joint_positions_rad=np.asarray(joint_positions[name])[:7],
                                 finger_positions_m=np.asarray(joint_positions[name])[7:])
                for name in ROBOT_NAMES}

    def _camera_pose(self, name, robot_state_at_frame):
        if name == 'scene_camera':
            return self._T_workcell_from_scene_cv.copy()
        arm = WRIST_ROBOTS[name]
        position, rotation = self._solvers[arm].compute_forward_kinematics(
            'panda_hand', np.array(robot_state_at_frame[arm].joint_positions_rad, copy=True))
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if position.shape != (3,) or rotation.shape != (3, 3) or not (
                np.isfinite(position).all() and np.isfinite(rotation).all()):
            raise RuntimeError(f'Invalid panda_hand FK for {arm}')
        T_workcell_from_hand = np.eye(4)
        T_workcell_from_hand[:3, :3] = rotation
        T_workcell_from_hand[:3, 3] = position
        return T_workcell_from_hand @ self._T_hand_from_camera_cv[name]

    def _unavailable(self, status, latest_robot_state, assembled_time_s, reason):
        self.last_errors = {name: reason for name in FORMAL_CAMERA_NAMES}
        return ObservationPacket(
            cameras={name: None for name in FORMAL_CAMERA_NAMES},
            camera_status={name: status for name in FORMAL_CAMERA_NAMES},
            latest_robot_state=latest_robot_state, assembled_time_s=assembled_time_s)

    def read(self, refresh=False):
        """Return three named views; a missing or stale view never becomes an old OK frame.

        refresh=True may render repeatedly without a physics step, so a new
        render sequence is not necessarily a new physical observation time.
        """
        assembled_time = float(self.env.time)
        latest = self._robot_states(self.env.observe_robot_joint_positions(), assembled_time)
        try:
            raw = self.env.observe_cameras(refresh=refresh, tolerate_errors=True,
                                           include_privileged=False)
        except (RuntimeError, ValueError) as exc:
            return self._unavailable('INVALID', latest, assembled_time, str(exc))
        raw_frames = raw.get('cameras', {})
        raw_errors = raw.get('camera_errors', {})
        if not raw_frames and not raw_errors:
            return self._unavailable('MISSING', latest, assembled_time, 'No active camera frames')

        system = self.env.camera_system
        rendered_joints = system.rendered_joint_positions
        sample_time = raw.get('simulation_time_s')
        sample_tick = raw.get('physics_frame_index')
        sequence_id = raw.get('render_frame_index')
        if (rendered_joints is None or sample_time is None or sample_tick is None
                or sequence_id is None or sample_tick != system.rendered_tick
                or not math.isclose(float(sample_time), system.rendered_time,
                                    rel_tol=0, abs_tol=1e-9)):
            return self._unavailable('INVALID', latest, assembled_time,
                                     'Camera frame has no matching measured joint snapshot')
        sample_time = float(sample_time)
        age_s = assembled_time - sample_time
        if not math.isfinite(age_s) or age_s < -1e-9:
            return self._unavailable('INVALID', latest, assembled_time,
                                     'Camera frame time is ahead of the robot clock')
        if age_s > self.max_frame_age_s:
            return self._unavailable('STALE', latest, assembled_time,
                                     f'Camera frame is {age_s:.3f} s old in simulation time')
        try:
            matched = self._robot_states(rendered_joints, sample_time)
        except (TypeError, ValueError) as exc:
            return self._unavailable('INVALID', latest, assembled_time, str(exc))

        frames, statuses, errors = {}, {}, {}
        for name in FORMAL_CAMERA_NAMES:
            if name in raw_errors:
                frames[name], statuses[name] = None, 'INVALID'
                errors[name] = raw_errors[name]
                continue
            data = raw_frames.get(name)
            if data is None:
                frames[name], statuses[name] = None, 'MISSING'
                errors[name] = 'Formal camera frame is absent'
                continue
            try:
                # Construct only from this allowlist. In particular, do not copy
                # instance_segmentation, instance_info, extrinsics or camera_params.
                rgb = np.asarray(data['rgb'])
                depth = np.asarray(data['depth'])
                valid_depth = np.isfinite(depth) & (depth > 0)
                if not valid_depth.any():
                    raise ValueError('Camera has no finite positive depth pixels')
                frames[name] = CameraFrame(
                    name=name, rgb=rgb, depth_m=depth, valid_depth=valid_depth,
                    K=np.asarray(data['intrinsics'], dtype=float),
                    T_workcell_from_camera_cv=self._camera_pose(name, matched),
                    sample_time_s=sample_time, sequence_id=sequence_id,
                    robot_state_at_frame=matched)
                statuses[name] = 'OK'
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                frames[name], statuses[name] = None, 'INVALID'
                errors[name] = str(exc)
        self.last_errors = errors
        return ObservationPacket(cameras=frames, camera_status=statuses,
                                 latest_robot_state=latest, assembled_time_s=assembled_time)