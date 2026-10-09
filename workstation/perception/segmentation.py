"""Instance masks on original RGB pixels; model scores are not grasp probabilities."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InstanceMask:
    mask: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label: str = 'cardboard box'

    def __post_init__(self):
        mask = np.array(self.mask, dtype=bool, copy=True)
        if mask.ndim != 2 or not mask.any():
            raise ValueError('Instance mask must be nonempty HxW')
        box = np.asarray(self.bbox_xyxy, dtype=float)
        h, w = mask.shape
        if (box.shape != (4,) or not np.isfinite(box).all() or
                not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
            raise ValueError('Bounding box is not in original-image pixels')
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError('Invalid uncalibrated detection score')
        mask.setflags(write=False)
        object.__setattr__(self, 'mask', mask)


@dataclass(frozen=True)
class SegmentationFrame:
    camera: str
    sequence_id: int
    time_s: float
    shape_hw: tuple[int, int]
    instances: tuple[InstanceMask, ...]
    source: str
    inference_ms: float = 0.

    def validate_for(self, frame):
        if (self.camera != frame.name or self.sequence_id != frame.sequence_id or
                abs(self.time_s-frame.sample_time_s) > 1e-9 or
                tuple(self.shape_hw) != frame.depth_m.shape or
                any(i.mask.shape != frame.depth_m.shape for i in self.instances)):
            raise ValueError('Segmentation is not aligned with the RGB-D exposure')
        return self
