"""Convert model proposals to robot candidates without projecting their rotation.

Known dimensions bound the visual cuboid; they are not fabricated point-cloud
samples. The simulator's robot model must still check every resulting path.
"""
import hashlib
import json
import numpy as np
from dataclasses import replace
from workstation.models.graspgen_client import tcp_poses,validate_grasps
from workstation.observations.camera_geometry import quaternion_from_matrix,pose_matrix
from workstation.planning.selection import GraspCandidate
from workstation.perception.scene_estimate import pose_is_observed


def upright_body_rotation(obj):
    c,s=np.cos(obj.axis_yaw_rad),np.sin(obj.axis_yaw_rad)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def opening_from_visual_box(tcp,obj):
    """Panda TCP Y is the closing axis; check an unmodified symmetric grasp."""
    centre=(np.asarray(obj.position_m)-tcp[:3,3])@tcp[:3,:3]
    radii=np.abs(tcp[:3,:3].T@upright_body_rotation(obj))@(np.asarray(obj.size_m)/2)
    margin=float(obj.uncertainty_m)
    width=2*(abs(centre[1])+radii[1]+margin+.003)
    closing=max(0.,2*radii[1]-.004)
    reasons=[]
    if not np.isfinite(width) or width>.080:reasons.append('GRIPPER_SPACE_INSUFFICIENT')
    # Do not silently recenter a model grasp to make it look feasible.
    if abs(centre[1])>.004:reasons.append('GRASP_OFF_CENTRE')
    return width,closing,dict(projected_span_m=float(2*radii[1]),
        centre_offset_m=float(centre[1]),visual_uncertainty_m=margin,
        source='current visual cuboid orientation + known size; not a model output'),tuple(reasons)


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
                        axis_yaw_rad=obj.axis_yaw_rad,size_m=obj.size_m,posture=obj.posture),
                    model=result['metadata'],calibration_sha256=digest,opening=evidence,
                    model_gripper_geometry_equivalent=bool(calibration.get('gripper_geometry_equivalence_verified')),
                    geometry_policy='model proposes only; validate with project robot model, not model mesh',
                    T_workcell_from_graspgen_base=base.tolist(),supported_skill='pose_preflight_only'),
                rejection_reasons=reasons))
    return output


def revalidate_candidate(candidate,obj,scene,sequence_ids):
    """Retain the ORIGINAL world pose only when fresh visual geometry agrees."""
    original=candidate.metadata['source_visual_object']
    if (obj.track_id!=candidate.target_id or obj.failures or obj.posture!='UPRIGHT' or
            not pose_is_observed(obj,scene) or obj.valid_depth_ratio<.9 or obj.uncertainty_m>.008):
        raise RuntimeError('MODEL_TARGET_UNCERTAIN')
    age=scene.assembled_time_s-obj.time_s
    if (not 0<=age<=.15 or abs(scene.observation_time_s-obj.time_s)>1e-6 or
            obj.time_s<=candidate.observation_time_s):
        raise RuntimeError('MODEL_TARGET_STALE')
    delta=float(np.linalg.norm(np.asarray(obj.position_m)-original['position_m']))
    angle=abs((obj.axis_yaw_rad-original['axis_yaw_rad']+np.pi/2)%np.pi-np.pi/2)
    if (delta>.002 or angle>np.deg2rad(3) or not np.allclose(obj.size_m,original['size_m'],atol=1e-6,rtol=0) or
            obj.velocity_m_s is None or np.linalg.norm(obj.velocity_m_s)>.015):
        raise RuntimeError('MODEL_TARGET_CHANGED')
    pose=pose_matrix(candidate.position_m,candidate.orientation_wxyz)
    width,closing,evidence,reasons=opening_from_visual_box(pose,obj)
    if reasons:raise RuntimeError('MODEL_GRASP_INVALID_ON_NEW_FRAME:'+reasons[0])
    return replace(candidate,validated_time_s=obj.time_s,opening_m=width,closing_width_m=closing,
        uncertainty_m=obj.uncertainty_m,metadata={**candidate.metadata,
            'revalidation':dict(time_s=obj.time_s,sequence_ids=sequence_ids,position_change_m=delta,
                axis_change_rad=float(angle),original_pose_retained=True,opening=evidence)})


def finger_preload_target(measured_positions,closing_width_m,preload_m=.004):
    """Bounded position preload after visual closure; NEVER holding evidence."""
    measured=np.asarray(measured_positions,float)
    if (measured.shape!=(2,) or not np.isfinite(measured).all() or
            np.any(measured<0) or np.any(measured>.0405) or
            not np.isfinite(closing_width_m) or not 0<=closing_width_m<=.08 or
            not np.isfinite(preload_m) or not 0<preload_m<=.004):
        raise ValueError('Invalid measured Panda preload request')
    return max(0.,min(closing_width_m/2,float(measured.min())-preload_m))
