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
    carton_size_catalog_m: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self):
        if not np.isfinite(self.optical_depth_error_margin_m) or self.optical_depth_error_margin_m<=0:
            raise ValueError('Optical depth error margin must be finite and positive')
        for size in self.carton_size_catalog_m:
            if len(size)!=3 or not np.isfinite(size).all() or min(size)<=0:
                raise ValueError('Carton catalogue requires positive finite dimensions')

    @classmethod
    def from_config(cls, config):
        # Explicit allowlist. No per-object initial positions, IDs or live states.
        return cls(tuple(config['boxes']['size']), float(config['table']['top_z']),
                   (-.38, .38, -.51, .16),
                   float(config['policy_observation']['max_frame_age_s']),
                   float(config['policy_observation']['sync_tolerance_s']),
                   float(config.get('diagnostic_optical_depth_error_margin_m',.003)),
                   tuple(tuple(size) for size in config.get('perception',{}).get('carton_size_catalog_m',())))


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
    # Optional CURRENT independent cuboid fit. Semantic/partial-view fields
    # above remain separate; consumers must validate this proof and its masks.
    measured_cuboid: Mapping | None = None


def current_cuboid_measurement(obj,scene):
    """Validated current whole-instance geometry, separate from top semantics."""
    from workstation.perception.cuboid_pose import measured_cuboid_supported
    measured=getattr(obj,'measured_cuboid',None)
    if measured is None:return None
    now=scene.observation_time_s;refs=set(measured.get('mask_refs',()))
    if (scene.frame=='workcell' and 0<=scene.assembled_time_s-now<=.15 and abs(obj.time_s-now)<=1e-6 and
            measured_cuboid_supported(measured,now) and refs==set(obj.mask_refs) and
            all(ref in scene.masks and scene.camera_status.get(ref.split('/')[0])=='OK' for ref in refs) and
            np.allclose(measured['size_m'],obj.size_m,atol=1e-6,rtol=0)):
        return measured
    return None


def pose_is_observed(obj,scene=None):
    """A full top fit OR independently measured pose; never inferred visibility.

    Consumers still check failures, semantics and motion. Scene-less use is
    only for constructing a tracker from an already checked image-time object.
    """
    if obj.visibility=='COMPLETE_TOP':return True
    if obj.visibility!='OBSERVED_CUBOID':return False
    from workstation.perception.cuboid_pose import measured_cuboid_supported
    measured=(current_cuboid_measurement(obj,scene) if scene is not None else
              getattr(obj,'measured_cuboid',None))
    return bool(measured is not None and measured_cuboid_supported(measured,obj.time_s) and
        set(measured['mask_refs'])==set(obj.mask_refs) and
        np.allclose(measured['size_m'],obj.size_m,atol=1e-6,rtol=0) and
        np.allclose(measured['position_m'],obj.position_m,atol=1e-8,rtol=0) and
        measured['uncertainty_m']<=obj.uncertainty_m<=.008)


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
