"""Convert model proposals to robot candidates without projecting their rotation.

Known dimensions bound the visual cuboid; they are not fabricated point-cloud
samples. The simulator's robot model must still check every resulting path.
"""
import math
import hashlib
import json
import numpy as np
from dataclasses import replace
from workstation.models.graspgen_client import tcp_poses,validate_grasps
from workstation.observations.camera_geometry import quaternion_from_matrix,pose_matrix
from workstation.planning.selection import GraspCandidate
from workstation.perception.scene_estimate import pose_is_observed


def upright_body_rotation(obj):
    """Body geometry reference; SIDE keeps its unresolved top-direction labels."""
    if obj.posture=='SIDE' or (obj.visibility=='OBSERVED_CUBOID' and pose_is_observed(obj)):
        if not obj.hypotheses:raise ValueError('Observed cuboid requires orientation hypotheses')
        return pose_matrix([0.,0.,0.],obj.hypotheses[0].orientation_wxyz)[:3,:3]
    c,s=np.cos(obj.axis_yaw_rad),np.sin(obj.axis_yaw_rad)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def opening_from_visual_box(tcp,obj):
    """Panda TCP Y is the closing axis; check an unmodified symmetric grasp."""
    centre=(np.asarray(obj.position_m)-tcp[:3,3])@tcp[:3,:3]
    radii=np.abs(tcp[:3,:3].T@upright_body_rotation(obj))@(np.asarray(obj.size_m)/2)
    margin=float(obj.uncertainty_m)
    # Prefer 3 mm extra clearance per finger. Near the opening limit, a
    # complete upright footprint may use 1.5 mm residual clearance beyond
    # the full visual uncertainty envelope. The gripper still opens to 80 mm.
    envelope=2*(abs(centre[1])+radii[1]+margin)
    minimum_clearance=.0015 if obj.posture=='UPRIGHT' and obj.visibility=='COMPLETE_TOP' else .003
    width=min(.080,envelope+.006) if envelope+2*minimum_clearance<=.080 else envelope+.006
    closing=max(0.,2*radii[1]-.004)
    reasons=[]
    if not np.isfinite(width) or width>.080:reasons.append('GRIPPER_SPACE_INSUFFICIENT')
    # Do not silently recenter a model grasp to make it look feasible.
    if abs(centre[1])>.004:reasons.append('GRASP_OFF_CENTRE')
    return width,closing,dict(projected_span_m=float(2*radii[1]),
        centre_offset_m=float(centre[1]),visual_uncertainty_m=margin,
        available_clearance_per_finger_m=float((.080-envelope)/2),
        source='current visual cuboid orientation + known size; not a model output'),tuple(reasons)


def refine_pinch_contact(candidate,obj):
    """Fit a nearby model pinch to the observed parallel sides and pad depth."""
    from workstation.observations.camera_geometry import pose_matrix,quaternion_from_matrix
    opening=candidate.metadata['opening']
    depth=opening.get('estimated_tip_engagement_m')
    direction=np.asarray(candidate.approach_direction)
    if (depth is None or depth<0 or -direction[2]<.98 or set(candidate.rejection_reasons)-
            {'FINGERTIPS_ABOVE_GRASP_BAND','GRASP_OFF_CENTRE','GRIPPER_SPACE_INSUFFICIENT'}):
        return candidate
    travel=min(.012,(.40*obj.size_m[2]-depth)/-direction[2]) if depth<.30*obj.size_m[2] else 0.
    pose=pose_matrix(candidate.position_m,candidate.orientation_wxyz)
    yaw_delta=0.
    lateral=0.
    if set(candidate.rejection_reasons)&{'GRASP_OFF_CENTRE','GRIPPER_SPACE_INSUFFICIENT'}:
        body_long=np.array([math.cos(obj.axis_yaw_rad),math.sin(obj.axis_yaw_rad),0.])
        tangent=pose[:3,0]
        if tangent@body_long<0:body_long=-body_long
        axis=pose[:3,2]
        yaw_delta=math.atan2(axis@np.cross(tangent,body_long),tangent@body_long)
        if abs(yaw_delta)>math.radians(12):return candidate
        c,s=math.cos(yaw_delta),math.sin(yaw_delta)
        pose[:3,:3]=pose[:3,:3]@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        lateral=float((np.asarray(obj.position_m)-pose[:3,3])@pose[:3,1])
        if abs(lateral)>.008:return candidate
        pose[:3,3]+=lateral*pose[:3,1]
    tangent=pose[:3,0]
    eccentricity=float((np.asarray(obj.position_m)-pose[:3,3])@tangent)
    slide=float(np.clip(eccentricity,-.020,.020)) if abs(eccentricity)>.012 else 0.
    if travel==0 and slide==0 and yaw_delta==0 and lateral==0:return candidate
    pose[:3,3]+=direction*travel+tangent*slide
    width,closing,evidence,reasons=opening_from_visual_box(pose,obj)
    raw=pose_matrix(candidate.position_m,candidate.orientation_wxyz)
    engagement=depth-(pose[2,3]-raw[2,3])+(abs(raw[2,1])*candidate.opening_m-abs(pose[2,1])*width)/2
    if reasons or engagement<.30*obj.size_m[2]:return candidate
    evidence['estimated_tip_engagement_m']=engagement
    orientation=tuple(quaternion_from_matrix(pose[:3,:3])) if yaw_delta else candidate.orientation_wxyz
    return replace(candidate,position_m=tuple(pose[:3,3]),orientation_wxyz=orientation,
        yaw_rad=math.atan2(pose[1,0],pose[0,0]),opening_m=width,closing_width_m=closing,
        rejection_reasons=(),metadata={**candidate.metadata,'opening':evidence,
        'model_tcp_before_seating_m':list(candidate.position_m),'axial_seating_m':travel,'tangent_slide_m':slide,
        'pad_alignment_rad':yaw_delta,'lateral_adjustment_m':lateral})


def candidates_from_result(result,obj,arms,calibration,sequence_ids):
    if result['metadata'].get('gripper_name')!='franka_panda':
        raise ValueError('Expected the verified Panda service')
    if abs(result['observation_time_s']-obj.time_s)>1e-6:
        raise ValueError('Candidate/visual target timestamp mismatch')
    poses,scores=validate_grasps(result['base_poses_workcell'],result['scores'])
    transformed=tcp_poses(poses,calibration['T_graspgen_base_from_project_tcp'],
                           calibration_verified=calibration.get('verified') is True)
    digest=hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest()
    output=[]
    for index,(pose,base,score) in enumerate(zip(transformed,poses,scores)):
        width,closing,evidence,reasons=opening_from_visual_box(pose,obj)
        if obj.posture=='UPRIGHT' and 'fingertip_reach_past_tcp_m' in calibration:
            tip_z=pose[2,3]+pose[2,2]*calibration['fingertip_reach_past_tcp_m']
            # Use the higher of the two inner fingertips for a tilted hand.
            tip_z+=abs(pose[2,1])*width/2
            engagement=float(obj.position_m[2]+obj.size_m[2]/2-tip_z)
            evidence['estimated_tip_engagement_m']=engagement
            if engagement<obj.uncertainty_m+.002:
                reasons+=('FINGERTIPS_ABOVE_GRASP_BAND',)
        # GraspGen supplies base +Z approach. Full TCP quaternion is retained.
        for arm in arms:
            output.append(GraspCandidate(obj.track_id,arm,tuple(pose[:3,3]),
                float(np.arctan2(pose[1,0],pose[0,0])),float(width),obj.uncertainty_m,
                orientation_wxyz=tuple(quaternion_from_matrix(pose[:3,:3])),source='graspgen',
                candidate_id=f"{result['request_id']}:{index}:{arm}",
                observation_time_s=result['observation_time_s'],
                approach_direction=tuple(base[:3,2]),closing_width_m=float(closing),
                opening_source=evidence['source'],model_score=float(score),
                metadata=dict(request_id=result['request_id'],sequence_ids=sequence_ids,frame='workcell',
                    source_visual_object=dict(track_id=obj.track_id,time_s=obj.time_s,position_m=obj.position_m,
                        axis_yaw_rad=obj.axis_yaw_rad,size_m=obj.size_m,posture=obj.posture,
                        body_rotation=upright_body_rotation(obj).tolist()),
                    model=result['metadata'],calibration_sha256=digest,opening=evidence,
                    model_gripper_geometry_equivalent=bool(calibration.get('gripper_geometry_equivalence_verified')),
                    geometry_policy='model proposes only; validate with project robot model, not model mesh',
                    T_workcell_from_graspgen_base=base.tolist(),supported_skill='pose_preflight_only'),
                rejection_reasons=reasons))
    return output


def revalidate_candidate(candidate,obj,scene,sequence_ids,accepted_postures=('UPRIGHT',)):
    """Retain the ORIGINAL world pose only when fresh visual geometry agrees."""
    original=candidate.metadata['source_visual_object']
    if (obj.track_id!=candidate.target_id or obj.failures or obj.posture not in accepted_postures or
            obj.posture!=original['posture'] or
            not pose_is_observed(obj,scene) or obj.valid_depth_ratio<.9 or obj.uncertainty_m>.008):
        raise RuntimeError('MODEL_TARGET_UNCERTAIN')
    age=scene.assembled_time_s-obj.time_s
    if (not 0<=age<=.15 or abs(scene.observation_time_s-obj.time_s)>1e-6 or
            obj.time_s<=candidate.observation_time_s):
        raise RuntimeError('MODEL_TARGET_STALE')
    delta=float(np.linalg.norm(np.asarray(obj.position_m)-original['position_m']))
    angle=abs((obj.axis_yaw_rad-original['axis_yaw_rad']+np.pi/2)%np.pi-np.pi/2)
    body_angle=None
    if 'body_rotation' in original:
        before=np.asarray(original['body_rotation']);current=upright_body_rotation(obj)
        signs=((1.,1.,1.),(-1.,-1.,1.)) if obj.posture=='UPRIGHT' else (
            (1.,1.,1.),(-1.,-1.,1.),(1.,-1.,-1.),(-1.,1.,-1.))
        body_angle=min(float(np.arccos(np.clip((np.trace(before.T@current@np.diag(s))-1)/2,-1.,1.))) for s in signs)
    if (delta>.002 or angle>np.deg2rad(3) or not np.allclose(obj.size_m,original['size_m'],atol=1e-6,rtol=0) or
            (body_angle is not None and body_angle>np.deg2rad(3)) or
            obj.velocity_m_s is None or np.linalg.norm(obj.velocity_m_s)>.015):
        raise RuntimeError('MODEL_TARGET_CHANGED')
    pose=pose_matrix(candidate.position_m,candidate.orientation_wxyz)
    width,closing,evidence,reasons=opening_from_visual_box(pose,obj)
    if reasons:raise RuntimeError('MODEL_GRASP_INVALID_ON_NEW_FRAME:'+reasons[0])
    return replace(candidate,validated_time_s=obj.time_s,opening_m=width,closing_width_m=closing,
        uncertainty_m=obj.uncertainty_m,metadata={**candidate.metadata,
            'revalidation':dict(time_s=obj.time_s,sequence_ids=sequence_ids,position_change_m=delta,
                axis_change_rad=float(angle),body_rotation_change_rad=body_angle,
                original_pose_retained=True,opening=evidence)})


def finger_preload_target(measured_positions,closing_width_m,preload_m=.004):
    """Bounded position preload after visual closure; NEVER holding evidence."""
    measured=np.asarray(measured_positions,float)
    if (measured.shape!=(2,) or not np.isfinite(measured).all() or
            np.any(measured<0) or np.any(measured>.0405) or
            not np.isfinite(closing_width_m) or not 0<=closing_width_m<=.08 or
            not np.isfinite(preload_m) or not 0<preload_m<=.004):
        raise ValueError('Invalid measured Panda preload request')
    return max(0.,min(closing_width_m/2,float(measured.min())-preload_m))
