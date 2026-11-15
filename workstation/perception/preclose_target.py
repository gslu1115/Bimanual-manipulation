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
    @classmethod
    def from_identity(cls,obj,identity,priors):
        """Use confirmed task knowledge ONLY as a correspondence/search hint."""
        if (obj.identity_id!=identity.identity_id or obj.size_m!=identity.size_m or
                obj.failures or obj.posture!='UPRIGHT' or not pose_is_observed(obj)):
            raise ValueError('Identity hint must come from confirmed geometry')
        target=object.__new__(cls)
        target.obj,target.priors,target.rotation=obj,priors,upright_body_rotation(obj)
        return target

    def __init__(self,obj,scene,priors):
        if (obj.failures or obj.posture!='UPRIGHT' or not pose_is_observed(obj,scene)
                or obj.uncertainty_m>.008 or abs(obj.time_s-scene.observation_time_s)>1e-6):
            raise ValueError('Pre-close reference requires a current confirmed upright target')
        if any(other.track_id!=obj.track_id and set(other.mask_refs).intersection(obj.mask_refs)
               for other in scene.objects):
            raise ValueError('Pre-close reference has an ambiguous neighbour')
        self.obj,self.priors=obj,priors
        self.rotation=upright_body_rotation(obj)

    def confirm(self,packet,scene,*,closed=False,refresh_complete=False):
        age=packet.observation_time_s-self.obj.time_s
        proof=dict(supported=False,reference_time_s=self.obj.time_s,
            reference_age_s=age,time_s=packet.observation_time_s,target_id=self.obj.track_id,
            reference_position_m=list(self.obj.position_m),phase='closed' if closed else 'approach',
            assumption='target remains stationary in the one hidden planar direction during bounded approach/close',
            full_pose_measured=False,grasp_proven=False)
        if not refresh_complete and not 0<=age<=6.:
            return dict(proof,reason='PRECLOSE_REFERENCE_EXPIRED')
        hint=getattr(self,'original',self) if refresh_complete else self
        measured=fit_cuboid_pose(packet,scene,hint.obj.position_m,hint.rotation,self.obj.size_m,self.priors,
                                allow_reference_translation=True,measure_side_interiors=True)
        proof['surface']=measured
        # A new independent measurement has its own timestamp. A fit using
        # the hidden-axis prior retains the old timestamp and the SAME 6 s limit.
        if refresh_complete and measured_cuboid_supported(measured,packet.observation_time_s):
            proof.update(reference_time_s=packet.observation_time_s,reference_age_s=0.)
        elif not 0<=age<=6.:
            return dict(proof,reason='PRECLOSE_REFERENCE_EXPIRED')
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

    def evaluate(self,packet,scene,*,closed=False):
        """One geometry check for pregrasp, approach and closure; no reclassification.

        Identity/size are task knowledge. Current contradictory posture or
        movement still invalidates the action. Only independently measured
        coordinates refresh the complete reference.
        """
        proof=self.confirm(packet,scene,closed=closed,refresh_complete=True)
        proof['next_action']=('CONTINUE' if proof['supported'] else
            'REPLAN' if proof['reason']=='PRECLOSE_VISIBLE_TARGET_MOVED' else 'OBSERVE')
        if not proof['supported'] or not proof['full_pose_measured'] or closed:return self,proof
        m=proof['surface'];r=np.asarray(m['rotation'])
        obj=replace(self.obj,time_s=packet.observation_time_s,position_m=tuple(m['position_m']),
            axis_yaw_rad=math.atan2(r[1,0],r[0,0]),mask_refs=tuple(m['mask_refs']),
            hypotheses=(PoseHypothesis('UPRIGHT',tuple(quaternion_from_matrix(r)),
                'confirmed face identity with current complete geometry'),),
            cameras=tuple(sorted({ref.split('/')[0] for ref in m['mask_refs']})),
            visibility='OBSERVED_CUBOID',uncertainty_m=m['uncertainty_m'],measured_cuboid=m,
            velocity_m_s=None)
        new=object.__new__(PreCloseTarget)
        new.obj,new.priors,new.rotation=obj,self.priors,r
        new.original=getattr(self,'original',self)
        # Compare subsequent fits with the selected pick, not a drifting chain.
        new.rotation=new.original.rotation
        return new,proof

    def renewed(self,scene,proof):
        """Refresh geometry age only from a full fit consistent with the original pick."""
        if (proof.get('phase')!='approach' or proof.get('target_id')!=self.obj.track_id or
                not preclose_supported(proof,scene.observation_time_s) or not proof['full_pose_measured']):
            return self
        original=getattr(self,'original',self)
        measured=proof['surface']
        rotation=np.asarray(measured['rotation'])
        angle=np.arccos(np.clip((np.trace(original.rotation.T@rotation)-1)/2,-1,1))
        if np.linalg.norm(np.asarray(measured['position_m'])-original.obj.position_m)>.002 or angle>math.radians(3):
            return self
        matches=[o for o in scene.objects if o.track_id==self.obj.track_id and o.visibility=='OBSERVED_CUBOID']
        if len(matches)!=1:return self
        obj=matches[0]
        obj=replace(obj,posture='UPRIGHT',hypotheses=tuple(h for h in obj.hypotheses if h.label=='UPRIGHT'))
        try:new=PreCloseTarget(obj,scene,self.priors)
        except ValueError:return self
        new.original=original
        return new

    def reacquire(self,packet,scene):
        """Rebuild a reference from current complete geometry, then join its masks.

        Proximity and clipped per-view catalogue choices are proposals, not
        identity proofs. Old pose/size only seed correspondence; all translation
        axes, size/support consistency and packaging need current evidence.
        """
        original=getattr(self,'original',self)
        obj=original.obj
        near=[other for other in scene.objects if
              np.linalg.norm(np.asarray(other.position_m)-obj.position_m)<.07]
        # Select size using actual top elevation on the calibrated pickup
        # support. A cropped footprint may fit the next smaller catalogue entry.
        sizes=self.priors.carton_size_catalog_m or (self.priors.carton_size_m,)
        height_supported=[]
        for current in near:
            height=current.quality.get('observed_top_z_m',float('nan'))-self.priors.support_z_m
            matching=[size for size in sizes if abs(size[2]-height)<=.002]
            if (len(matching)==1 and np.allclose(matching[0],obj.size_m,rtol=0,atol=1e-6) and
                    abs(current.time_s-packet.observation_time_s)<=1e-6):height_supported.append(current)
        if not height_supported:return None
        measured=fit_cuboid_pose(packet,scene,obj.position_m,original.rotation,obj.size_m,self.priors,
                                measure_side_interiors=True)
        if (not measured_cuboid_supported(measured,packet.observation_time_s) or
                measured['translation_difference_from_hint_m']>.002 or
                measured['orientation_difference_from_hint_deg']>3. or
                len({ref.split('/')[0] for ref in measured['mask_refs']})<2):
            return None
        refs=set(measured['mask_refs'])
        members=[o for o in near if o.mask_refs and set(o.mask_refs)<=refs]
        recoverable={'CAMERA_SIZE_CONFLICT','PARTIAL_OR_MERGED_GEOMETRY',
                     'INCOMPATIBLE_SIZE_OR_SUPPORT','INSTANCE_GEOMETRY_INCONSISTENT','VERTICAL_EXTENT_UNRESOLVED'}
        if (not members or not any(o in members for o in height_supported) or
                {ref for o in members for ref in o.mask_refs}!=refs or
                sum(len(o.mask_refs) for o in members)!=len(refs) or
                sum(not o.failures for o in members)>1 or
                any(not set(o.failures)<=recoverable or o.quality.get('packaging_conflict',0) or
                    o.posture in ('INVERTED','SIDE') or abs(o.time_s-packet.observation_time_s)>1e-6
                    for o in members) or
                any(o.track_id==obj.track_id and o not in members for o in scene.objects)):
            return None
        current=min(members,key=lambda o:np.linalg.norm(np.asarray(o.position_m)-obj.position_m))
        bounded=replace(current,position_m=tuple(measured['position_m']),size_m=obj.size_m,
            size_source='catalogue size verified by current top elevation and two-view cuboid fit',
            mask_refs=tuple(measured['mask_refs']),cameras=tuple(sorted({ref.split('/')[0] for ref in refs})),
            failures=(),measured_cuboid=measured,velocity_m_s=None)
        from workstation.perception.rgbd_cartons import CartonEstimator
        resolved=CartonEstimator(self.priors)._resolve_measured_flat_cuboid(packet,scene,bounded,measured,obj)
        if (resolved.posture=='UNKNOWN' and not resolved.failures and len(members)==1 and
                current.posture=='UPRIGHT' and not current.failures):
            # Preserve an already valid CURRENT semantic observation. Conflicted
            # or fragmented detections must instead resolve packaging above.
            resolved=replace(resolved,posture='UPRIGHT',
                hypotheses=tuple(h for h in resolved.hypotheses if h.label=='UPRIGHT'))
        if resolved.posture!='UPRIGHT' or resolved.failures:return None
        return replace(resolved,velocity_m_s=None,quality=dict(resolved.quality,
            preclose_reacquired_source_tracks=float(len(members)),
            preclose_resolved_size_conflict=float(any('CAMERA_SIZE_CONFLICT' in o.failures for o in members))))


def reconcile_reacquired_target(estimator,scene,current):
    """Remove only complete masks covered by this new independent fit."""
    from workstation.perception.scene_estimate import current_cuboid_measurement
    candidate=replace(scene,objects=(current,))
    if current_cuboid_measurement(current,candidate) is None or current.posture!='UPRIGHT' or current.failures:
        raise RuntimeError('PRECLOSE_REACQUISITION_PROOF_INVALID')
    merged=estimator.associate_supported_target(scene,current.track_id,current.measured_cuboid)
    matches=[o for o in merged.objects if o.track_id==current.track_id and set(o.mask_refs)==set(current.mask_refs)]
    if len(matches)!=1:raise RuntimeError('PRECLOSE_REACQUISITION_ASSOCIATION_UNRESOLVED')
    estimator._tracks[current.track_id]=current
    estimator._anchors[current.track_id]=current
    return replace(merged,objects=tuple(current if o.track_id==current.track_id else o for o in merged.objects))


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
