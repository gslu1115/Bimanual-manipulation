"""Checks of formal camera packets; renderer poses stay in diagnostics."""
from __future__ import annotations
import math
import numpy as np
from workstation.observations.observation_packet import FORMAL_CAMERA_NAMES


def check(env):
    report = dict(success=False, scope="policy_input_interface_only", frames={}, motion_checks={})
    for _ in range(8): env.step(monitor_conveyor=False)

    def frame(label):
        packet = env.observe_policy_inputs(refresh=True)
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
        print("[sensor-check] "+label, flush=True)
        return transforms

    before = frame("home")
    for robot, camera, other, change, label in (
            ("panda_left", "left_wrist_camera", "right_wrist_camera", .15, "left_moved"),
            ("panda_right", "right_wrist_camera", "left_wrist_camera", -.15, "both_moved")):
        target = env.observe_robot_joint_positions()[robot]
        target[0] += change
        env.command_joints(robot, target)
        for _ in range(60): env.step(monitor_conveyor=False)
        after = frame(label)
        movement = float(np.max(np.abs(after[camera]-before[camera])))
        other_movement = float(np.max(np.abs(after[other]-before[other])))
        scene_movement = float(np.max(np.abs(after["scene_camera"]-before["scene_camera"])))
        if movement < .02 or other_movement > .005 or scene_movement > 1e-6:
            raise RuntimeError(f"Camera motion independence failed: {robot}")
        report["motion_checks"][robot] = dict(wrist_transform_change=movement,
            other_wrist_transform_change=other_movement, scene_transform_change=scene_movement)
        before = after

    stale_steps = math.ceil(env._policy_sensor_adapter.max_frame_age_s / env.c["physics"]["dt"]) + 1
    for _ in range(stale_steps): env.step(monitor_conveyor=False, render=False)
    report["stale_test_physics_steps"] = stale_steps
    packet = env.observe_policy_inputs()
    if set(packet.camera_status.values()) != {"STALE"} or any(
            value is not None for value in packet.cameras.values()):
        raise RuntimeError("Stale camera frames were not removed")
    report["stale"] = dict(packet.camera_status)
    if set(env.observe_policy_inputs(refresh=True).camera_status.values()) != {"OK"}:
        raise RuntimeError("Refresh did not restore the camera packets")

    removed = env.camera_system.sensors.pop("right_wrist_camera")
    try:
        packet = env.observe_policy_inputs()
        if dict(packet.camera_status) != dict(
                scene_camera="OK", left_wrist_camera="OK", right_wrist_camera="MISSING"):
            raise RuntimeError("A missing camera affected other views")
        report["missing"] = dict(packet.camera_status)
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
