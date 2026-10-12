"""Convert model proposals to robot candidates without projecting their rotation.

Known dimensions bound the visual cuboid; they are not fabricated point-cloud
samples. The simulator's robot model must still check every resulting path.
"""
import hashlib
import json
import numpy as np
from workstation.models.graspgen_client import tcp_poses,validate_grasps
from workstation.observations.camera_geometry import quaternion_from_matrix
from workstation.planning.selection import GraspCandidate


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
                    model=result['metadata'],calibration_sha256=digest,opening=evidence,
                    model_gripper_geometry_equivalent=bool(calibration.get('gripper_geometry_equivalence_verified')),
                    geometry_policy='model proposes only; validate with project robot model, not model mesh',
                    T_workcell_from_graspgen_base=base.tolist(),supported_skill='pose_preflight_only'),
                rejection_reasons=reasons))
    return output
