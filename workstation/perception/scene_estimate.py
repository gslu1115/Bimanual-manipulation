"""Minimal, explicit visual scene contract. Scores are diagnostics, not probabilities."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ScenePriors:
    carton_size_m: tuple[float, float, float]
    support_z_m: float
    workspace_xy: tuple[float, float, float, float]
    max_age_s: float = .15
    sync_tolerance_s: float = .02
    optical_depth_error_margin_m: float = .003

    def __post_init__(self):
        if not np.isfinite(self.optical_depth_error_margin_m) or self.optical_depth_error_margin_m<=0:
            raise ValueError('Optical depth error margin must be finite and positive')

    @classmethod
    def from_config(cls, config):
        # Explicit allowlist. No per-object initial positions, IDs or live states.
        return cls(tuple(config['boxes']['size']), float(config['table']['top_z']),
                   (-.38, .38, -.51, .16),
                   float(config['policy_observation']['max_frame_age_s']),
                   float(config['policy_observation']['sync_tolerance_s']),
                   float(config.get('diagnostic_optical_depth_error_margin_m',.003)))


@dataclass(frozen=True)
class PoseHypothesis:
    label: str
    orientation_wxyz: tuple[float, float, float, float]
    evidence: str


@dataclass(frozen=True)
class ObjectEstimate:
    track_id: str
    time_s: float
    position_m: tuple[float, float, float]
    axis_yaw_rad: float
    size_m: tuple[float, float, float]
    size_source: str
    posture: str
    hypotheses: tuple[PoseHypothesis, ...]
    mask_refs: tuple[str, ...]
    cameras: tuple[str, ...]
    visibility: str
    valid_depth_ratio: float
    quality: Mapping[str, float]
    failures: tuple[str, ...]
    velocity_m_s: tuple[float, float, float] | None = None
    uncertainty_m: float = .005


@dataclass(frozen=True)
class ObservedEdge:
    camera: str
    mask_ref: str
    kind: str
    pixels_uv: tuple[tuple[int, int], tuple[int, int]]
    endpoints_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    evidence: str
    uncertainty_m: float = .003


@dataclass(frozen=True)
class SceneEstimate:
    observation_time_s: float
    assembled_time_s: float
    frame: str
    objects: tuple[ObjectEstimate, ...]
    failures: tuple[str, ...]
    camera_status: Mapping[str, str]
    masks: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    edges: tuple[ObservedEdge, ...] = ()
    schema: str = 'scene_estimate_v1'

    def to_dict(self):
        return dict(schema=self.schema, observation_time_s=self.observation_time_s,
                    assembled_time_s=self.assembled_time_s, frame=self.frame,
                    objects=[asdict(obj) for obj in self.objects], failures=self.failures,
                    camera_status=dict(self.camera_status), mask_refs=list(self.masks),
                    edges=[asdict(edge) for edge in self.edges])
