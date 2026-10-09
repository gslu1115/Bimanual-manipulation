"""Checks of formal camera packets; renderer poses stay in diagnostics."""
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from workstation.observations.observation_packet import (FORMAL_CAMERA_NAMES, POLICY_CAMERA_NAMES,
    POLICY_STATE_SCHEMA, PANDA_ARM_JOINT_NAMES, PANDA_FINGER_JOINT_NAMES, PandaJointMap)
from workstation.observations.policy_observation import PolicyObservationAdapter


def wait_robot_rest(env, timeout_s=3., min_s=.5, hold_s=.1, max_arm_speed_rad_s=.02):
    """Wait for measured settling; a fixed step count can mislabel residual motion."""
    dt, stable = env.c['physics']['dt'], 0.
    for step in range(math.ceil(timeout_s/dt)):
        env.step(monitor_conveyor=False, render=False)
        states = env.observe_robot_states()
        speed = max(float(np.max(np.abs(state.joint_velocities_rad_s))) for state in states.values())
        stable = stable+dt if speed <= max_arm_speed_rad_s else 0.
        elapsed = (step+1)*dt
        if elapsed >= min_s and stable >= hold_s:
            return dict(physics_steps=step+1, elapsed_s=elapsed, max_arm_speed_rad_s=speed,
                        speed_tolerance_rad_s=max_arm_speed_rad_s, stable_hold_s=stable)
    raise RuntimeError(f'Robots did not settle within {timeout_s} s; last speed={speed:.6f} rad/s')


def check(env):
    policy_adapter = PolicyObservationAdapter.from_config(env.c)
    report = dict(success=False, scope="camera_robot_policy_interface_only", frames={}, motion_checks={},
                  policy_frames={}, rest_checks={},
                  policy_state_schema=list(POLICY_STATE_SCHEMA), robot_dof_mapping={})
    for side, robot in (("left", "panda_left"), ("right", "panda_right")):
        mapping = PandaJointMap(tuple(env.arms[robot].dof_names))
        offset = 0 if side == "left" else 8
        report["robot_dof_mapping"][robot] = dict(dof_names=list(mapping.dof_names),
            arm_dof_indices=mapping.indices(PANDA_ARM_JOINT_NAMES),
            finger_dof_indices=mapping.indices(PANDA_FINGER_JOINT_NAMES),
            arm_policy_slots=list(range(offset, offset+7)), gripper_policy_slot=offset+7)
    report['rest_checks']['home'] = wait_robot_rest(env)

    def frame(label):
        packet = env.get_observation_packet(refresh=True)
        if set(packet.camera_status.values()) != {"OK"}:
            raise RuntimeError(f"Unavailable policy frames: {dict(packet.camera_status)}")
        raw = env.observe_cameras(refresh=False)  # Independent diagnostic reference only.
        details, transforms = {}, {}
        for name in FORMAL_CAMERA_NAMES:
            observed = packet.cameras[name]
            actual = observed.T_workcell_from_camera_cv
            reference = np.asarray(raw["cameras"][name]["extrinsics"]["T_world_from_camera_opencv"])
            position_error = float(np.linalg.norm(actual[:3, 3]-reference[:3, 3]))
            rotation_error = math.degrees(math.acos(float(np.clip(
                (np.trace(actual[:3, :3].T@reference[:3, :3])-1)/2, -1, 1))))
            if position_error >= .005 or rotation_error >= .5:
                raise RuntimeError(f"Camera FK/calibration mismatch: {name}")
            if not math.isclose(observed.sample_time_s, env.time, abs_tol=1e-9):
                raise RuntimeError(f"Unexpected frame time: {name}")
            details[name] = dict(rgb_shape=list(observed.rgb.shape),
                depth_shape=list(observed.depth_m.shape),
                positive_depth_fraction=float(observed.valid_depth.mean()),
                translation_error_m=position_error, rotation_error_deg=rotation_error,
                sample_time_s=observed.sample_time_s, sequence_id=observed.sequence_id)
            transforms[name] = actual.copy()
        report["frames"][label] = details
        policy = policy_adapter(packet)
        if (policy["state"].shape != (16,) or policy["state"].dtype != np.float32
                or not policy["sync_ok"] or not all(policy["image_mask"].values())):
            raise RuntimeError("Policy state/mask/synchronization check failed")
        robots = {}
        for side, state in packet.robot.items():
            if state.qvel is None or state.qvel.shape != (9,):
                raise RuntimeError(f"Missing measured joint velocities: {side}")
            if state.tcp_pose_world is None or state.tcp_pose_world.shape != (7,):
                raise RuntimeError(f"Missing encoder-derived Lula TCP: {side}")
            robots[side] = dict(arm_qpos_shape=list(state.joint_positions_rad.shape),
                arm_qvel_shape=list(state.joint_velocities_rad_s.shape),
                finger_qpos_shape=list(state.finger_positions_m.shape),
                finger_qvel_shape=list(state.finger_velocities_m_s.shape),
                qpos=state.qpos.tolist(), qvel=state.qvel.tolist(),
                gripper_width_m=state.gripper_width_m, timestamp=state.sample_time_s,
                tcp_pose_world=None if state.tcp_pose_world is None else state.tcp_pose_world.tolist())
        stacked = policy_adapter.stack_rgb(policy)
        flat = policy_adapter.to_lerobot_dict(policy)
        expected = (3, *policy["images"]["cam_high"].shape)
        if stacked.shape != expected or len(flat) != 4:
            raise RuntimeError("ACT/LeRobot formatting check failed")
        report["policy_frames"][label] = dict(timestamp=policy["timestamp"], robots=robots,
            image_shapes={name: list(policy["images"][name].shape) for name in POLICY_CAMERA_NAMES},
            image_mask=policy["image_mask"], state_shape=list(policy["state"].shape),
            state_dtype=str(policy["state"].dtype), state=policy["state"].tolist(),
            stack_rgb_shape=list(stacked.shape), sync_ok=policy["sync_ok"],
            metadata=policy["metadata"], lerobot_keys=list(flat))
        print(f"[sensor-check] {label}: timestamp={packet.timestamp:.6f}, "
              f"state={policy['state'].shape} {policy['state'].dtype}, "
              f"stack={stacked.shape}, masks={list(policy['image_mask'].values())}, "
              f"sync_ok={policy['sync_ok']}", flush=True)
        for side, state in packet.robot.items():
            print(f"  {side}: arm qpos/qvel={state.joint_positions_rad.shape}/"
                  f"{state.joint_velocities_rad_s.shape}, finger qpos/qvel="
                  f"{state.finger_positions_m.shape}/{state.finger_velocities_m_s.shape}, "
                  f"gripper_width={state.gripper_width_m:.6f} m", flush=True)
        for name in POLICY_CAMERA_NAMES:
            observed = packet.cameras[name]
            print(f"  {name}: RGB={observed.rgb.shape} {observed.rgb.dtype}, "
                  f"depth={observed.depth_m.shape} {observed.depth_m.dtype}, valid={observed.valid}", flush=True)
        skew_states = {name: replace(state, sample_time_s=state.sample_time_s+
            policy_adapter.sync_tolerance_s+.01) for name, state in packet.robot_state.items()}
        if policy_adapter(replace(packet, robot_state=skew_states))["sync_ok"]:
            raise RuntimeError("Injected camera/robot timestamp skew was not detected")
        report["timestamp_skew_fault_injection"] = dict(success=True,
            injected_offset_s=policy_adapter.sync_tolerance_s+.01, sync_ok=False)
        env._write_json("sensor_check_report.json", report)  # Preserve partial evidence on a later failure.
        return transforms

    before = frame("home")
    for robot, camera, other, change, label in (
            ("panda_left", "left_wrist_camera", "right_wrist_camera", .15, "left_moved"),
            ("panda_right", "right_wrist_camera", "left_wrist_camera", -.15, "both_moved")):
        target = env.observe_robot_joint_positions()[robot]
        joint1 = PandaJointMap(tuple(env.arms[robot].dof_names)).indices(('panda_joint1',))[0]
        target[joint1] += change
        env.command_joints(robot, target)
        report['rest_checks'][label] = wait_robot_rest(env)
        after = frame(label)
        movement = float(np.max(np.abs(after[camera]-before[camera])))
        other_movement = float(np.max(np.abs(after[other]-before[other])))
        scene_movement = float(np.max(np.abs(after["scene_camera"]-before["scene_camera"])))
        report["motion_checks"][robot] = dict(wrist_transform_change=movement,
            other_wrist_transform_change=other_movement, scene_transform_change=scene_movement)
        env._write_json("sensor_check_report.json", report)
        if movement < .02 or other_movement > .005 or scene_movement > 1e-6:
            raise RuntimeError(f"Camera motion independence failed: {robot}; "
                               f"moving={movement}, other={other_movement}, scene={scene_movement}")
        before = after

    stale_steps = math.ceil(env._policy_sensor_adapter.max_frame_age_s / env.c["physics"]["dt"]) + 1
    for _ in range(stale_steps): env.step(monitor_conveyor=False, render=False)
    report["stale_test_physics_steps"] = stale_steps
    packet = env.observe_policy_inputs()
    if set(packet.camera_status.values()) != {"STALE"} or any(
            value is not None for value in packet.cameras.values()):
        raise RuntimeError("Stale camera frames were not removed")
    report["stale"] = dict(packet.camera_status)
    if any(policy_adapter(packet)["image_mask"].values()):
        raise RuntimeError("Stale images were given valid policy masks")
    if set(env.observe_policy_inputs(refresh=True).camera_status.values()) != {"OK"}:
        raise RuntimeError("Refresh did not restore the camera packets")

    removed = env.camera_system.sensors.pop("right_wrist_camera")
    try:
        packet = env.observe_policy_inputs()
        if dict(packet.camera_status) != dict(
                scene_camera="OK", left_wrist_camera="OK", right_wrist_camera="MISSING"):
            raise RuntimeError("A missing camera affected other views")
        report["missing"] = dict(packet.camera_status)
        missing = policy_adapter(packet)
        if missing["image_mask"]["cam_right_wrist"] or missing["images"]["cam_right_wrist"].any():
            raise RuntimeError("Missing camera has no zero policy placeholder")
    finally:
        env.camera_system.sensors["right_wrist_camera"] = removed

    class FailedAnnotator:
        def get_data(self):
            raise RuntimeError("Injected single-camera acquisition failure")
    annotators = env.camera_system.sensors["left_wrist_camera"]["annotators"]
    original = annotators["rgb"]
    try:
        annotators["rgb"] = FailedAnnotator()
        packet = env.observe_policy_inputs()
        if dict(packet.camera_status) != dict(
                scene_camera="OK", left_wrist_camera="INVALID", right_wrist_camera="OK"):
            raise RuntimeError("A failed camera affected other views")
        report["single_error"] = dict(packet.camera_status)
        invalid = policy_adapter(packet)
        if invalid["image_mask"]["cam_left_wrist"] or invalid["images"]["cam_left_wrist"].any():
            raise RuntimeError("Invalid camera has no zero policy placeholder")
    finally:
        annotators["rgb"] = original

    settings, saved_settings = env.camera_system.settings, env.camera_system.settings.copy()
    try:
        settings.update(save_images=False, recording_enabled=True,
                        render_interval_steps=12, capture_interval_steps=24)
        renders = env.camera_system.metrics["render_calls"]
        saves = env.camera_system.metrics["saved_frames"]
        previous_images = {path for path in (env.output/"recorded").rglob("*")
                           if path.suffix in (".png", ".npy")}
        for _ in range(48): env.step(monitor_conveyor=False)
        renders = env.camera_system.metrics["render_calls"]-renders
        saves = env.camera_system.metrics["saved_frames"]-saves
        images = [str(path.relative_to(env.output)) for path in (env.output/"recorded").rglob("*")
                  if path.suffix in (".png", ".npy") and path not in previous_images]
        if renders != 4 or saves != 2 or images:
            raise RuntimeError("Low-frequency recording produced unexpected captures or images")
        report["low_frequency_no_image_save"] = dict(success=True, physics_steps=48,
            render_calls=renders, periodic_captures=saves, image_files=images)
    finally:
        settings.clear()
        settings.update(saved_settings)
    report["success"] = True
    env._write_json("sensor_check_report.json", report)
    return report
