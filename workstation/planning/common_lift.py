"""Select a connected pregrasp IK branch for a pair of measured grasp poses."""
from itertools import product

import numpy as np

from workstation.planning.collision import sphere_arrays
from workstation.planning.plan_reuse import CandidatePrefixCache


PAIR_MARGIN_M = .008


def mutual_gap(robot, specs):
    spheres = [sphere_arrays(robot.spheres(spec.candidate.arm, spec.lift_q)) for spec in specs]
    (_, left, lr), (_, right, rr) = spheres
    return float((np.linalg.norm(left[:, None] - right, axis=2) - lr[:, None] - rr).min())


def elbow_offset(robot, arm, q, partner):
    """Signed elbow distance toward the outside, relative to the shoulder-wrist line."""
    axis = np.asarray(robot.base_config[arm]['base_xyz'],float) - robot.base_config[partner]['base_xyz']
    axis[2] = 0.
    axis /= np.linalg.norm(axis)
    shoulder,elbow,wrist=(robot.fk(arm,q,f)[0] for f in ('panda_link2','panda_link4','panda_link7'))
    reach=wrist-shoulder
    reference=shoulder+reach*np.clip((elbow-shoulder)@reach/max(reach@reach,1e-12),0.,1.)
    return float((elbow-reference)@axis)


def outward_pregrasps(robot, spec, pregrasp, partner):
    arm = spec.candidate.arm
    baseline = elbow_offset(robot,arm,pregrasp,partner)
    pick = np.asarray(spec.candidate.position_m)
    goal = pick - np.asarray(spec.candidate.approach_direction) * spec.candidate.pregrasp_distance_m
    solutions = []
    # J3 swivels the redundant elbow. Evaluate both signs in workcell
    # coordinates since Panda base orientations and grasp poses differ.
    def solve(seed):
        try:
            q = robot.solve_pose(arm, goal, spec.candidate.orientation_wxyz,
                                 np.clip(seed, robot.limits[0]+.005, robot.limits[1]-.005))
        except RuntimeError:
            return
        shift = elbow_offset(robot,arm,q,partner) - baseline
        if shift > .001 and all(np.max(np.abs(q-other)) >= .015 for _,other in solutions):
            solutions.append((shift,q))
    for delta in (.65, -.65, 1.3, -1.3):
        seed=pregrasp.copy()
        seed[2]+=delta
        solve(seed)
    if solutions:
        # Continue along the IK branch that actually improved its elbow offset.
        # Re-solve the original TCP pose, never alter the grasp or transition
        # between distant branches while carrying a carton.
        direction=max(solutions,key=lambda item:item[0])[1]-pregrasp
        for scale in (2.,4.):
            solve(pregrasp+scale*direction)
    return sorted(solutions, key=lambda row: -row[0])


def checked_variants(robot, scene, spec, path, partner):
    c = spec.candidate
    variants = []
    for shift, hint in outward_pregrasps(robot, spec, np.asarray(path[-1]), partner):
        cache = CandidatePrefixCache()
        result = robot.feasibility(c, spec.obj, scene, spec.slot, pregrasp_hint=hint,
                                   prefix_cache=cache, slot_endpoints=True)
        if result['reasons']:
            continue
        key = cache.key(robot, c, spec.obj, scene, hint, True)
        q, _, _, approach = cache.pregrasps[key]
        # solve_pregrasp may reject its hinted branch and try a standard seed.
        if np.max(np.abs(q-hint)) > .015:
            continue
        variants.append((cache.destinations[c.candidate_id,spec.slot], approach, shift))
    return variants


def common_lift_options(robot, scene, specs, paths, variant_cache, audit):
    gap = mutual_gap(robot,specs)
    audit.update(default_gap_m=gap, required_m=PAIR_MARGIN_M, alternatives_checked=0)
    def offset(index,spec):
        return elbow_offset(robot,spec.candidate.arm,spec.lift_q,specs[1-index].candidate.arm)
    original=[offset(i,s) for i,s in enumerate(specs)]
    audit['default_elbow_offsets_m']=original
    if gap >= PAIR_MARGIN_M:
        yield specs,paths,dict(audit,selected_gap_m=gap,outward_arms=[],selected_elbow_offsets_m=original)
        return
    alternatives = []
    for index,s in enumerate(specs):
        key = (s.candidate.candidate_id,s.slot)
        if key not in variant_cache:
            variant_cache[key] = checked_variants(robot,scene,s,paths[index],specs[1-index].candidate.arm)
        alternatives.append([(s,paths[index],0.,original[index])]+
            [(v,p,d,offset(index,v)) for v,p,d in variant_cache[key]])
    ranked=[]
    for pair in product(*alternatives):
        selected,approach,shifts,offsets=zip(*pair)
        if not any(shifts):continue
        clearance=mutual_gap(robot,selected)
        audit['alternatives_checked']+=1
        if clearance<PAIR_MARGIN_M:continue
        # Maximize the more inward of the two elbows first. Any feasible
        # both-outward posture ranks ahead of every posture with an inward elbow.
        detail=dict(selected_gap_m=clearance,selected_elbow_offsets_m=list(offsets),
                    outward_arms=[s.candidate.arm for s,d in zip(selected,shifts) if d],
                    pregrasp_elbow_outward_m=list(shifts))
        ranked.append(((min(offsets),clearance),selected,approach,detail))
    for _,selected,approach,detail in sorted(ranked,key=lambda row:row[0],reverse=True):
        yield selected,approach,dict(audit,**detail)
