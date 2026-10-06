"""Deployable three-camera observation contract.

The packet contains only data obtainable from physical RGB-D cameras and robot
encoders. Camera poses are calibrated workcell transforms, never renderer poses.
All timestamps in one packet must use the same clock.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np


FORMAL_CAMERA_NAMES = (
    "scene_camera",
    "left_wrist_camera",
    "right_wrist_camera",
)
ROBOT_NAMES = ("panda_left", "panda_right")
CAMERA_STATUSES = frozenset(("OK", "MISSING", "STALE", "INVALID"))


def _finite_time(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite, nonnegative time")
    return value


def _number_array(value: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have shape {shape} and finite values")
    result.setflags(write=False)
    return result


def _robot_states(value: Mapping[str, RobotState], label: str) -> Mapping[str, RobotState]:
    states = dict(value)
    if set(states) != set(ROBOT_NAMES):
        raise ValueError(f"{label} must contain exactly {ROBOT_NAMES}")
    if any(not isinstance(state, RobotState) for state in states.values()):
        raise TypeError(f"{label} values must be RobotState")
    return MappingProxyType(states)


@dataclass(frozen=True)
class RobotState:
    """Measured seven arm joints and two finger joints for one Panda.

    The timestamp is the robot-state sampling time in the packet's common clock.
    Joint positions are radians; individual finger positions are metres.
    """

    sample_time_s: float
    joint_positions_rad: np.ndarray
    finger_positions_m: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_time_s",
                           _finite_time(self.sample_time_s, "RobotState.sample_time_s"))
        object.__setattr__(self, "joint_positions_rad",
                           _number_array(self.joint_positions_rad, (7,), "joint_positions_rad"))
        object.__setattr__(self, "finger_positions_m",
                           _number_array(self.finger_positions_m, (2,), "finger_positions_m"))

    @property
    def gripper_width_m(self) -> float:
        """Opening inferred from the two measured finger positions."""
        return float(np.sum(self.finger_positions_m))


@dataclass(frozen=True)
class CameraFrame:
    """One RGB-D exposure paired with robot state from that exposure time.

    RGB and depth share one rectified pixel grid; K refers to that grid.
    T_workcell_from_camera_cv maps camera optical coordinates (+X right, +Y down,
    +Z forward) to metres in the workcell frame. sequence_id is a generic camera
    frame number; it can come from the renderer or a physical camera driver.
    A hardware adapter must use the image acquisition timestamp as sample_time_s.
    """

    name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    valid_depth: np.ndarray
    K: np.ndarray
    T_workcell_from_camera_cv: np.ndarray
    sample_time_s: float
    sequence_id: int
    robot_state_at_frame: Mapping[str, RobotState]

    def __post_init__(self) -> None:
        if self.name not in FORMAL_CAMERA_NAMES:
            raise ValueError(f"CameraFrame.name must be one of {FORMAL_CAMERA_NAMES}")

        rgb = np.asarray(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be a uint8 HxWx3 image")
        rgb = rgb.copy()
        rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)

        depth = np.asarray(self.depth_m)
        if not np.issubdtype(depth.dtype, np.floating) or depth.shape != rgb.shape[:2]:
            raise ValueError("depth_m must be a floating-point HxW array matching rgb")
        depth = depth.copy()
        depth.setflags(write=False)
        object.__setattr__(self, "depth_m", depth)

        valid = np.asarray(self.valid_depth)
        if valid.dtype != np.bool_ or valid.shape != rgb.shape[:2]:
            raise ValueError("valid_depth must be a boolean HxW array matching rgb")
        if not np.all(np.isfinite(depth[valid]) & (depth[valid] > 0)):
            raise ValueError("valid_depth marks nonfinite or nonpositive depth")
        valid = valid.copy()
        valid.setflags(write=False)
        object.__setattr__(self, "valid_depth", valid)

        K = _number_array(self.K, (3, 3), "K")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], (0, 0, 1), atol=1e-6):
            raise ValueError("K must have positive focal lengths and a homogeneous last row")
        object.__setattr__(self, "K", K)

        transform = _number_array(self.T_workcell_from_camera_cv, (4, 4),
                                  "T_workcell_from_camera_cv")
        if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-6):
            raise ValueError("T_workcell_from_camera_cv must be homogeneous")
        rotation = transform[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)):
            raise ValueError("T_workcell_from_camera_cv must contain a proper rotation")
        object.__setattr__(self, "T_workcell_from_camera_cv", transform)

        object.__setattr__(self, "sample_time_s",
                           _finite_time(self.sample_time_s, "CameraFrame.sample_time_s"))
        if isinstance(self.sequence_id, (bool, np.bool_)) or not isinstance(
                self.sequence_id, (int, np.integer)) or self.sequence_id < 0:
            raise ValueError("sequence_id must be a nonnegative integer")
        object.__setattr__(self, "sequence_id", int(self.sequence_id))

        states = _robot_states(self.robot_state_at_frame, "robot_state_at_frame")
        if any(not math.isclose(state.sample_time_s, self.sample_time_s,
                                rel_tol=0, abs_tol=1e-6) for state in states.values()):
            raise ValueError("robot_state_at_frame must be aligned to the exposure time")
        object.__setattr__(self, "robot_state_at_frame", states)


@dataclass(frozen=True)
class ObservationPacket:
    """One assembled observation, with explicit availability for all three views."""

    cameras: Mapping[str, CameraFrame | None]
    camera_status: Mapping[str, str]
    latest_robot_state: Mapping[str, RobotState]
    assembled_time_s: float

    def __post_init__(self) -> None:
        cameras = dict(self.cameras)
        statuses = dict(self.camera_status)
        if set(cameras) != set(FORMAL_CAMERA_NAMES) or set(statuses) != set(FORMAL_CAMERA_NAMES):
            raise ValueError("cameras and camera_status must name exactly the three formal cameras")
        for name in FORMAL_CAMERA_NAMES:
            frame, status = cameras[name], statuses[name]
            if status not in CAMERA_STATUSES:
                raise ValueError(f"Invalid camera status for {name}: {status}")
            if frame is not None and (not isinstance(frame, CameraFrame) or frame.name != name):
                raise ValueError(f"Camera frame for {name} has the wrong name or type")
            if status == "OK" and frame is None:
                raise ValueError(f"Camera {name} is OK without a frame")
            if status != "OK" and frame is not None:
                raise ValueError(f"Camera {name} is {status} but has a frame")
        object.__setattr__(self, "cameras", MappingProxyType(cameras))
        object.__setattr__(self, "camera_status", MappingProxyType(statuses))
        object.__setattr__(self, "latest_robot_state",
                           _robot_states(self.latest_robot_state, "latest_robot_state"))
        object.__setattr__(self, "assembled_time_s",
                           _finite_time(self.assembled_time_s, "ObservationPacket.assembled_time_s"))
