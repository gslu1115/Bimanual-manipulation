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


def outward_pregrasps(robot, spec, pregrasp, partner):
    arm = spec.candidate.arm
    axis = np.subtract(robot.base_config[arm]['base_xyz'], robot.base_config[partner]['base_xyz'])
    axis[2] = 0.
    axis /= np.linalg.norm(axis)
    def elbow(q):
        return float(robot.fk(arm, q, 'panda_link4')[0] @ axis)
    baseline = elbow(pregrasp)
    pick = np.asarray(spec.candidate.position_m)
    goal = pick - np.asarray(spec.candidate.approach_direction) * spec.candidate.pregrasp_distance_m
    solutions = []
    # J3 swivels the redundant elbow. Evaluate both signs in workcell
    # coordinates since Panda base orientations and grasp poses differ.
    for delta in (.65, -.65, 1.3, -1.3):
        seed = pregrasp.copy()
        seed[2] += delta
        try:
            q = robot.solve_pose(arm, goal, spec.candidate.orientation_wxyz,
                                 np.clip(seed, robot.limits[0]+.005, robot.limits[1]-.005))
        except RuntimeError:
            continue
        shift = elbow(q) - baseline
        if shift > .001 and all(np.max(np.abs(q-other)) >= .015 for _,other in solutions):
            solutions.append((shift,q))
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
    if gap >= PAIR_MARGIN_M:
        yield specs,paths,dict(audit,selected_gap_m=gap,outward_arms=[])
        return
    defaults = [(s,p,0.) for s,p in zip(specs,paths)]
    alternatives = []
    def try_option(options):
        selected, approach, shifts = zip(*options)
        clearance = mutual_gap(robot, selected)
        audit['alternatives_checked'] += 1
        if clearance >= PAIR_MARGIN_M:
            yield selected,approach,dict(audit,selected_gap_m=clearance,
                outward_arms=[s.candidate.arm for s,d in zip(selected,shifts) if d],
                pregrasp_elbow_outward_m=list(shifts))
    for index,s in enumerate(specs):
        key = (s.candidate.candidate_id,s.slot)
        if key not in variant_cache:
            variant_cache[key] = checked_variants(robot,scene,s,paths[index],specs[1-index].candidate.arm)
        variations = variant_cache[key]
        alternatives.append(variations)
        for option in variations:
            pair = defaults.copy()
            pair[index] = option
            yield from try_option(pair)
    for pair in product(*alternatives):
        yield from try_option(pair)
