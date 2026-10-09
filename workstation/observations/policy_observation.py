"""NumPy-only policy formatting; acquisition and controllers stay independent."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from workstation.observations.observation_packet import (
    POLICY_CAMERA_NAMES, POLICY_CAMERA_SOURCES, POLICY_STATE_SCHEMA, ROBOT_NAMES,
    ObservationPacket,
)

STATE_SCHEMA_VERSION = "panda_positions_and_width_v1"
DEFAULTS = dict(camera_names=list(POLICY_CAMERA_NAMES), state_schema=STATE_SCHEMA_VERSION,
                sync_tolerance_s=0.02, max_frame_age_s=0.15, include_depth=False)


def policy_settings(config):
    settings = {**DEFAULTS, **config.get("policy_observation", {})}
    if tuple(settings["camera_names"]) != POLICY_CAMERA_NAMES:
        raise ValueError(f"policy_observation.camera_names must be {POLICY_CAMERA_NAMES} in that order")
    if settings["state_schema"] != STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported policy state schema: {settings['state_schema']}")
    for key in ("sync_tolerance_s", "max_frame_age_s"):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"policy_observation.{key} must be finite and nonnegative")
    if settings["max_frame_age_s"] <= 0:
        raise ValueError("max_frame_age_s must be positive")
    if type(settings["include_depth"]) is not bool:
        raise ValueError("policy_observation.include_depth must be boolean")
    return settings


class PolicyObservationAdapter:
    """Produce three separate CHW images and the explicit 16-dimensional V1 state.

    Shapes are supplied in (height, width), including for absent camera frames.
    State comes from packet.robot_state at observation time, never latest_robot_state.
    Consumers must inspect sync_ok and image_mask before requesting actions.
    """

    def __init__(self, camera_shapes: Mapping[str, tuple[int, int]], *,
                 sync_tolerance_s=0.02, max_frame_age_s=0.15, include_depth=False):
        if set(camera_shapes) != set(POLICY_CAMERA_NAMES):
            raise ValueError("camera_shapes must define exactly the three semantic camera names")
        self.camera_shapes = {name: tuple(camera_shapes[name]) for name in POLICY_CAMERA_NAMES}
        if any(len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)
               for shape in self.camera_shapes.values()):
            raise ValueError("Camera shapes must be positive integer (height, width)")
        settings = policy_settings({"policy_observation": dict(sync_tolerance_s=sync_tolerance_s,
            max_frame_age_s=max_frame_age_s, include_depth=include_depth)})
        self.sync_tolerance_s = float(settings["sync_tolerance_s"])
        self.max_frame_age_s = float(settings["max_frame_age_s"])
        self.include_depth = settings["include_depth"]

    @classmethod
    def from_config(cls, config):
        settings = policy_settings(config)
        cameras = {cam["name"]: cam for cam in config["cameras"]}
        shapes = {name: tuple(reversed(cameras[source]["resolution"]))
                  for name, source in POLICY_CAMERA_SOURCES.items()}
        return cls(shapes, sync_tolerance_s=settings["sync_tolerance_s"],
                   max_frame_age_s=settings["max_frame_age_s"], include_depth=settings["include_depth"])

    def __call__(self, packet: ObservationPacket):
        if not isinstance(packet, ObservationPacket):
            raise TypeError("PolicyObservationAdapter requires an ObservationPacket")
        images, masks, times, statuses, errors, depths, depth_masks = {}, {}, {}, {}, {}, {}, {}
        for name, source in POLICY_CAMERA_SOURCES.items():
            shape = self.camera_shapes[name]
            frame, status = packet.cameras[source], packet.camera_status[source]
            times[name] = None if frame is None else frame.sample_time_s
            valid = status == "OK" and frame is not None
            if valid and frame.rgb.shape != (*shape, 3):
                valid, status = False, "INVALID"
                errors[name] = "RGB resolution differs from configured policy shape"
            if valid:
                age = packet.assembled_time_s - frame.sample_time_s
                if age < -1e-9:
                    valid, status = False, "INVALID"
                    errors[name] = "Camera acquisition time is ahead of assembly clock"
                elif age > self.max_frame_age_s:
                    valid, status = False, "STALE"
                    errors[name] = "Camera exceeds configured maximum simulation-frame age"
            masks[name], statuses[name] = bool(valid), status
            images[name] = (np.ascontiguousarray(frame.rgb.transpose(2, 0, 1), dtype=np.float32)
                            / np.float32(255) if valid else np.zeros((3, *shape), np.float32))
            if self.include_depth:
                # Depth remains separate; missing pixels have a separate boolean mask.
                depths[name] = (np.where(frame.valid_depth, frame.depth_m, 0).astype(np.float32)
                                if valid else np.zeros(shape, np.float32))
                depth_masks[name] = frame.valid_depth.copy() if valid else np.zeros(shape, bool)

        states = packet.robot_state
        state = np.concatenate([
            np.concatenate((states[robot].joint_positions_rad, [states[robot].gripper_width_m]))
            for robot in ROBOT_NAMES
        ]).astype(np.float32)
        if state.shape != (len(POLICY_STATE_SCHEMA),) or not np.isfinite(state).all():
            raise ValueError("V1 policy state is not a finite float32 vector of length 16")
        robot_times = {robot: states[robot].sample_time_s for robot in ROBOT_NAMES}
        valid_times = [times[name] for name in POLICY_CAMERA_NAMES if masks[name]]
        camera_robot_skew = max((abs(t - r) for t in valid_times for r in robot_times.values()),
                                default=None)
        all_times = valid_times + list(robot_times.values()) + [packet.timestamp]
        span = max(all_times) - min(all_times)
        robot_age = {robot: packet.assembled_time_s - t for robot, t in robot_times.items()}
        robot_fresh = all(-1e-9 <= age <= self.max_frame_age_s for age in robot_age.values())
        sync_ok = bool(all(masks.values()) and robot_fresh and span <= self.sync_tolerance_s + 1e-9)
        result = dict(images=images, image_mask=masks, state=state, timestamp=packet.timestamp,
                      sync_ok=sync_ok, metadata=dict(camera_timestamps=times,
            robot_timestamps=robot_times, assembled_time_s=packet.assembled_time_s,
            camera_status=statuses, camera_errors=errors, sync_ok=sync_ok,
            max_camera_robot_skew_s=camera_robot_skew, timestamp_span_s=span,
            sync_tolerance_s=self.sync_tolerance_s, robot_fresh=robot_fresh,
            state_schema=POLICY_STATE_SCHEMA, state_schema_version=STATE_SCHEMA_VERSION))
        if self.include_depth:
            result.update(depths=depths, depth_mask=depth_masks)
        return result

    @staticmethod
    def stack_rgb(observation):
        return stack_rgb(observation)

    @staticmethod
    def to_lerobot_dict(observation):
        return to_lerobot_dict(observation)


def stack_rgb(observation):
    """ACT-style KxCxHxW in POLICY_CAMERA_NAMES order, without joining pixel grids."""
    images = [observation["images"][name] for name in POLICY_CAMERA_NAMES]
    if len({image.shape for image in images}) != 1:
        raise ValueError("stack_rgb requires equal camera resolutions; use the images dictionary")
    return np.stack(images, axis=0)


def to_lerobot_dict(observation):
    """Map four model fields only; reject invalid data since this form omits masks.

    A recorder must retain the original masks/timestamps separately and add its
    own action. This helper is neither a dataset writer nor an action interface.
    """
    if not observation["sync_ok"] or not all(observation["image_mask"].values()):
        raise ValueError("LeRobot model fields require synchronized valid views; reobserve first")
    return {**{f"observation.images.{name}": observation["images"][name]
               for name in POLICY_CAMERA_NAMES}, "observation.state": observation["state"]}
