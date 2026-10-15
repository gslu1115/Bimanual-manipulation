"""Short approach/close continuity from current surfaces and a confirmed target.

One hidden planar coordinate may remain conditional on the target being
stationary. Never creates a new full-pose measurement from the old pose and
never uses a grasp relation before closing.
"""
from dataclasses import replace
import math
import numpy as np

from workstation.perception.cuboid_pose import (fit_cuboid_pose, measured_cuboid_supported,
                                              reference_constrained_cuboid_supported)
from workstation.perception.scene_estimate import pose_is_observed,PoseHypothesis
from workstation.observations.camera_geometry import quaternion_from_matrix
from workstation.planning.graspgen_candidates import upright_body_rotation


class PreCloseTarget:
    def __init__(self,obj,scene,priors):
        if (obj.failures or obj.posture!='UPRIGHT' or not pose_is_observed(obj,scene)
                or obj.uncertainty_m>.008 or abs(obj.time_s-scene.observation_time_s)>1e-6):
            raise ValueError('Pre-close reference requires a current confirmed upright target')
        if any(other.track_id!=obj.track_id and np.linalg.norm(np.asarray(other.position_m)-obj.position_m)<.07
               for other in scene.objects):
            raise ValueError('Pre-close reference has an ambiguous neighbour')
        self.obj,self.priors=obj,priors
        self.rotation=upright_body_rotation(obj)

    def confirm(self,packet,scene,*,closed=False):
        age=packet.observation_time_s-self.obj.time_s
        proof=dict(supported=False,reference_time_s=self.obj.time_s,
            reference_age_s=age,time_s=packet.observation_time_s,target_id=self.obj.track_id,
            reference_position_m=list(self.obj.position_m),phase='closed' if closed else 'approach',
            assumption='target remains stationary in the one hidden planar direction during bounded approach/close',
            full_pose_measured=False,grasp_proven=False)
        if not 0<=age<=6.:
            return dict(proof,reason='PRECLOSE_REFERENCE_EXPIRED')
        measured=fit_cuboid_pose(packet,scene,self.obj.position_m,self.rotation,self.obj.size_m,self.priors,
                                allow_reference_translation=True)
        proof['surface']=measured
        if not (measured_cuboid_supported(measured,packet.observation_time_s) or
                reference_constrained_cuboid_supported(measured,packet.observation_time_s)):
            return dict(proof,reason=measured.get('reason','PRECLOSE_SURFACES_UNSUPPORTED'))
        refs=set(measured['mask_refs'])
        if any((obj.posture in ('INVERTED','SIDE') and not obj.failures or
                obj.quality.get('packaging_conflict',0)) and refs.intersection(obj.mask_refs)
               for obj in scene.objects):
            return dict(proof,reason='PRECLOSE_CURRENT_POSTURE_CONFLICT')
        # Opening checks retain the model's 2 mm / 3 degree consistency gate.
        # During closing, the existing held-geometry fitter allows up to 8 mm /
        # 8 degrees of measured settling; its new relation is latched only then.
        distance=.008 if closed else .002
        angle=8. if closed else 3.
        if (measured['translation_difference_from_hint_m']>distance or
                measured['orientation_difference_from_hint_deg']>angle):
            return dict(proof,reason='PRECLOSE_VISIBLE_TARGET_MOVED')
        if abs(measured['position_m'][2]-(self.priors.support_z_m+self.obj.size_m[2]/2))>.008:
            return dict(proof,reason='PRECLOSE_TARGET_NOT_ON_SUPPORT')
        return dict(proof,supported=True,reason='CURRENT_SURFACES_SUPPORT_CONFIRMED_TARGET',
            full_pose_measured=measured['geometric_pose_measured'])


def preclose_supported(proof,time_s):
    if proof.get('supported') is not True or not proof.get('target_id') or proof.get('grasp_proven') is not False:
        return False
    values=np.asarray([proof.get(k,float('nan')) for k in ('time_s','reference_time_s','reference_age_s')])
    if (not np.isfinite(values).all() or abs(values[0]-time_s)>1e-6 or
            not 0<=values[2]<=6. or abs(values[0]-values[1]-values[2])>1e-6):return False
    measured=proof.get('surface',{})
    if not (measured_cuboid_supported(measured,time_s) or reference_constrained_cuboid_supported(measured,time_s)):
        return False
    if proof.get('full_pose_measured') is not measured['geometric_pose_measured']:return False
    if proof.get('phase') not in ('approach','closed'):return False
    closed=proof['phase']=='closed'
    return (measured['translation_difference_from_hint_m']<=(.008 if closed else .002) and
            measured['orientation_difference_from_hint_deg']<=(8. if closed else 3.))


def reconcile_preclose_target(estimator,scene,proof):
    """Merge only masks supported by the current fit; retain all other obstacles."""
    if not preclose_supported(proof,scene.observation_time_s):
        raise RuntimeError('PRECLOSE_UNSUPPORTED_IDENTITY_PROOF')
    measured=proof['surface'];target_id=proof['target_id'];refs=set(measured['mask_refs'])
    scene=estimator.associate_supported_target(scene,target_id,measured)
    matches=[obj for obj in scene.objects if obj.track_id==target_id and set(obj.mask_refs)==refs]
    if len(matches)!=1:raise RuntimeError('PRECLOSE_FRAGMENT_ASSOCIATION_UNRESOLVED')
    original=matches[0];r=np.asarray(measured['rotation']);size=np.asarray(measured['size_m'])
    independent=measured['geometric_pose_measured']
    hypotheses=tuple(PoseHypothesis(label,tuple(quaternion_from_matrix(r@flip)),
        'current supported geometry; hidden top/bottom semantics unresolved') for label,flip in
        (('UPRIGHT',np.eye(3)),('INVERTED',np.diag([1.,-1.,-1.]))))
    bounded=replace(original,position_m=tuple(measured['position_m']),axis_yaw_rad=math.atan2(r[1,0],r[0,0]),
        size_m=tuple(size),posture='UNKNOWN',hypotheses=hypotheses,failures=(),velocity_m_s=None,
        visibility='OBSERVED_CUBOID' if independent else 'REFERENCE_CONSTRAINED_CUBOID',
        uncertainty_m=measured['uncertainty_m'],measured_cuboid=measured if independent else None,
        quality=dict(original.quality,geometric_length_m=float(size[0]),geometric_width_m=float(size[1]),
            observed_height_m=float(size[2]),preclose_reference_constrained=float(not independent),
            independent_cuboid_pose=float(independent)))
    estimator._tracks[target_id]=bounded
    # No refresh of the old complete/semantic anchor from a constrained pose.
    return replace(scene,objects=tuple(bounded if obj.track_id==target_id else obj for obj in scene.objects))
