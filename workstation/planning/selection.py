"""Finite grasp templates and skill routing; no simulator target state."""
from dataclasses import dataclass, asdict
import math
import numpy as np


# Upright side pinch: keep the palm above the carton top while retaining
# contact on the 45 mm tall side. Shared by ranking, refreshed targets and
# nominal held preflight; runtime uses the measured visual TCP relation.
UPRIGHT_TCP_HEIGHT_M = .004
# Controlled placement releases the BODY 10 mm above the known support.
# TCP/body offset is additional, not the release gap. Gravity/contact settles
# the released carton; independent visual verification is still mandatory.
UPRIGHT_RELEASE_GAP_M = .010
UPRIGHT_PLACE_TCP_CLEARANCE_M = UPRIGHT_TCP_HEIGHT_M + UPRIGHT_RELEASE_GAP_M


def upright_grasp_position(position_m):
    return np.asarray(position_m) + [0., 0., UPRIGHT_TCP_HEIGHT_M]


@dataclass(frozen=True)
class GraspCandidate:
    target_id: str
    arm: str
    position_m: tuple[float,float,float]
    yaw_rad: float
    opening_m: float
    uncertainty_m: float


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: GraspCandidate
    feasible: bool
    reasons: tuple[str,...]
    terms: dict
    score: float  # Uncalibrated rule utility, NEVER a success probability.


def assess(scene, robot_states, feasibility, home, max_age_s=.15,failure_history=None,priorities=None,eligible=None):
    """feasibility(candidate, object, scene) must check IK AND whole-path collisions."""
    results = []
    failure_history=failure_history or {}; priorities=priorities or {}
    for obj in scene.objects:
        if eligible is not None and not eligible(obj,scene):continue
        reasons = list(obj.failures)
        failures=int(failure_history.get(obj.track_id,0))
        if failures >= 3: reasons.append('RETRY_LIMIT_REACHED')
        if scene.assembled_time_s-obj.time_s > max_age_s:
            reasons.append('STALE_ESTIMATE')
        if obj.posture != 'UPRIGHT':
            reasons.append('SIDE_POSTURE' if obj.posture == 'SIDE' else
                           'INVERTED_POSTURE' if obj.posture == 'INVERTED' else 'POSTURE_UNCERTAIN')
        if obj.visibility != 'COMPLETE_TOP': reasons.append('OCCLUDED')
        if obj.valid_depth_ratio < .9: reasons.append('DEPTH_INSUFFICIENT')
        if obj.velocity_m_s is None:
            reasons.append('MOTION_NOT_OBSERVED')
        elif np.linalg.norm(obj.velocity_m_s) > .015:
            reasons.append('TARGET_MOVING')
        # A known-size grasp closes across the narrow top dimension.
        opening = obj.size_m[1]+.018
        if opening > .080: reasons.append('GRIPPER_SPACE_INSUFFICIENT')
        radius = np.linalg.norm(np.array(obj.size_m[:2]))/2+.040
        clearance = min((np.linalg.norm(np.array(obj.position_m[:2])-other.position_m[:2])
            -radius-np.linalg.norm(np.array(other.size_m[:2]))/2-other.uncertainty_m
            for other in scene.objects if other.track_id != obj.track_id),default=.2)
        if clearance < .008: reasons.append('NEIGHBOUR_SPACE_INSUFFICIENT')
        for arm,state in robot_states.items():
            for yaw in (obj.axis_yaw_rad,obj.axis_yaw_rad+math.pi):
                candidate = GraspCandidate(obj.track_id,arm,
                    tuple(upright_grasp_position(obj.position_m)),yaw,opening,obj.uncertainty_m)
                path = dict(reasons=(),joint_cost=0.,path_clearance_m=0.)
                if not reasons:
                    path = feasibility(candidate,obj,scene)
                all_reasons = tuple(dict.fromkeys(reasons+list(path['reasons'])))
                terms = dict(geometry_quality=float(obj.quality['footprint_coverage']),
                    valid_depth_ratio=obj.valid_depth_ratio,visibility_complete=float(obj.visibility == 'COMPLETE_TOP'),
                    uncertainty_m=obj.uncertainty_m, neighbour_clearance_m=float(clearance),
                    joint_cost=float(path['joint_cost']),path_clearance_m=float(path['path_clearance_m']),
                    task_priority=float(priorities.get(obj.track_id,1.)),previous_failures=failures,
                    whole_task_template_checked=bool(path.get('whole_task_template_checked',False)),
                    checked_phases=path.get('checked_phases',[]),joint_plans=path.get('joint_plans',{}),
                    whole_task_check_scope=path.get('whole_task_check_scope'),
                    pending_observation_phases=path.get('pending_observation_phases',[]),
                    pregrasp_joint_positions_rad=path.get('pregrasp_joint_positions_rad'))
                if all_reasons:
                    terms['rejection_phase']=path.get('rejection_phase')
                    terms['rejection_details']=path.get('rejection_details',{})
                utility = (2*terms['geometry_quality']+min(clearance,.10)*5
                           +min(terms['path_clearance_m'],.05)*5
                           +terms['task_priority']*.2+terms['valid_depth_ratio']*.3
                           -terms['joint_cost']*.2-obj.uncertainty_m*50-failures*.5)
                results.append(CandidateAssessment(candidate,not all_reasons,all_reasons,terms,utility))
    return sorted(results,key=lambda item:(not item.feasible,-item.score,item.candidate.target_id,item.candidate.arm))


def select_skill(assessments, available=('DIRECT_PICK_PLACE','REOBSERVE')):
    good = next((a for a in assessments if a.feasible),None)
    if good is not None:
        return dict(skill='DIRECT_PICK_PLACE',candidate=asdict(good.candidate),
                    utility=good.score,utility_is_probability=False,terms=good.terms)
    reasons = sorted({reason for a in assessments for reason in a.reasons}) or ['NO_TARGET']
    categories={reason:failure_category(reason) for reason in reasons}
    # Lack of reachability/path/space is never interpreted as inversion or side pose.
    desired = ('SIDE_ADJUST' if reasons == ['SIDE_POSTURE'] else
               'DUAL_ARM_FLIP' if reasons == ['INVERTED_POSTURE'] else 'REOBSERVE')
    return dict(skill=desired if desired in available else 'STOP_UNSUPPORTED_SKILL',
                desired_skill=desired,reasons=reasons,reason_categories=categories)


def failure_category(reason):
    if reason in ('UNOBSERVED_PATH_SPACE','MISSING_WORKSPACE_OBSERVATION'): return 'UNOBSERVED_SPACE'
    if reason in ('OBSERVED_PATH_OBSTACLE','OBSERVED_SLOT_OCCUPIED'): return 'PATH_OR_COLLISION'
    if reason == 'RETRY_LIMIT_REACHED': return 'RETRY_EXHAUSTED'
    if reason in ('SIDE_POSTURE','INVERTED_POSTURE'): return 'POSTURE_TEMPLATE_UNAVAILABLE'
    if 'SPACE_INSUFFICIENT' in reason: return 'GRIPPER_OR_NEIGHBOUR_SPACE'
    if reason.startswith(('ROBOT_UNREACHABLE','JOINT_LIMIT')): return 'ROBOT_REACHABILITY'
    if 'COLLISION' in reason or 'PATH_BLOCKED' in reason or reason == 'IK_DISCONTINUITY': return 'PATH_OR_COLLISION'
    if reason == 'OCCLUDED': return 'OCCLUSION'
    if reason == 'TARGET_MOVING': return 'OBSERVED_MOTION'
    return 'PERCEPTION_UNCERTAIN'
