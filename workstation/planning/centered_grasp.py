"""Explicit upright cuboid template, independent of learned grasp proposals."""
from dataclasses import replace
import math
import numpy as np
from workstation.observations.camera_geometry import pose_matrix, quaternion_from_matrix
from workstation.perception.scene_estimate import pose_is_observed
from workstation.planning.graspgen_candidates import opening_from_visual_box, revalidate_candidate, upright_body_rotation
from workstation.planning.selection import GraspCandidate


def centered_candidates(obj, arms, sequence_ids):
    """Pinch the measured long-axis midpoint, 12 mm below the top at TCP.

    Top-relative height preserves palm clearance across different box heights.
    This is a geometric proposal, not a contact/holding guarantee. Full robot
    path checks and current post-close visual proof remain mandatory.
    """
    position=np.asarray(obj.position_m)+[0.,0.,obj.size_m[2]/2-.012]
    rows=[]
    for side,yaw in enumerate((obj.axis_yaw_rad,obj.axis_yaw_rad+math.pi)):
        orientation=np.array([0.,math.cos(yaw/2),math.sin(yaw/2),0.])
        pose=pose_matrix(position,orientation)
        width,closing,evidence,reasons=opening_from_visual_box(pose,obj)
        for arm in arms:
            rows.append(GraspCandidate(obj.track_id,arm,tuple(position),yaw,width,obj.uncertainty_m,
                orientation_wxyz=tuple(quaternion_from_matrix(pose[:3,:3])),source='centered',
                candidate_id=f'centered:{obj.time_s}:{obj.track_id}:{side}:{arm}',
                observation_time_s=obj.time_s,closing_width_m=closing,opening_source=evidence['source'],
                metadata=dict(sequence_ids=sequence_ids,frame='workcell',
                    source_visual_object=dict(track_id=obj.track_id,time_s=obj.time_s,
                        position_m=obj.position_m,axis_yaw_rad=obj.axis_yaw_rad,size_m=obj.size_m,
                        posture=obj.posture,body_rotation=upright_body_rotation(obj).tolist()),
                    proposal='visual upright long-axis midpoint; TCP 12 mm below top; two opposite wrist orientations',
                    opening=evidence,supported_skill='upright_pick_place',model_inference=False),
                rejection_reasons=reasons))
    return rows


class CenteredGraspSource:
    backend='centered'

    def __init__(self):self.requests=[]

    def generate(self,packet,scene,robot,refresh):
        rows=[]
        for obj in scene.objects:
            if not obj.failures and obj.posture=='UPRIGHT' and pose_is_observed(obj,scene):
                rows.extend(centered_candidates(obj,robot.read_states(),
                    {n:f.sequence_id for n,f in packet.cameras.items() if f}))
        robot.hold(.025)
        packet,scene=refresh()
        checked=[]
        for row in rows:
            if not row.rejection_reasons:
                try:
                    matches=[o for o in scene.objects if o.track_id==row.target_id]
                    if len(matches)!=1:raise RuntimeError('MODEL_TARGET_UNCERTAIN')
                    row=revalidate_candidate(row,matches[0],scene,
                        {n:f.sequence_id for n,f in packet.cameras.items() if f})
                except RuntimeError as error:row=replace(row,rejection_reasons=(str(error),))
            checked.append(row)
        self.requests.append(dict(backend=self.backend,model_inference=False,
            fresh_observation_time_s=scene.observation_time_s,candidates=len(checked),
            supported_and_revalidated=sum(not c.rejection_reasons for c in checked)))
        return checked
