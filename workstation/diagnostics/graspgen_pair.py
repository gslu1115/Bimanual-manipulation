"""GraspGen proposals for two distinct visual tracks in one frozen Isaac scene."""
from dataclasses import asdict, replace
from collections import Counter
from itertools import product
import math
import time

import numpy as np

from workstation.perception.scene_estimate import pose_is_observed, ScenePriors
from workstation.planning.collision import check_spheres
from workstation.planning.graspgen_candidates import candidates_from_result, revalidate_candidate
from workstation.planning.selection import assess
from workstation.planning.task_targets import ConveyorSlots


def synchronized_knots(left, right, max_step=.025):
    """Preserve both path corners and sample a common normalized approach clock."""
    paths = [np.asarray(path, dtype=float) for path in (left, right)]
    if any(p.ndim != 2 or len(p) < 2 or p.shape[1] != 7 or not np.isfinite(p).all() for p in paths):
        raise ValueError('Expected two finite, seven-joint paths')
    clocks = [np.linspace(0., 1., len(p)) for p in paths]
    knots = np.unique(np.concatenate(clocks))
    aligned = [np.stack([np.interp(knots, t, col) for col in p.T], axis=1)
               for t, p in zip(clocks, paths)]
    yield aligned[0][0], aligned[1][0]
    for i in range(1, len(knots)):
        steps = max(1, math.ceil(max(np.max(np.abs(p[i]-p[i-1])) for p in aligned)/max_step))
        for n in range(1, steps+1):
            yield tuple(p[i-1]+(p[i]-p[i-1])*n/steps for p in aligned)


def paired_pregrasp(robot, rows, states, scene):
    arms = ('panda_left', 'panda_right')
    feasible = [[a for a in rows if a.feasible and a.candidate.source == 'graspgen'
                 and a.candidate.arm == arm and a.terms.get('full_pose_checked')] for arm in arms]
    pairs = [(left, right) for left, right in product(*feasible)
             if left.candidate.target_id != right.candidate.target_id
             and ('slot_xy' not in left.terms or left.terms['slot_xy'] != right.terms['slot_xy'])]
    pairs.sort(key=lambda pair: -(pair[0].score+pair[1].score))
    paths = {}
    rejected = []
    for pair in pairs:
        for arm, a in zip(arms, pair):
            c = a.candidate
            if c.candidate_id not in paths:
                try:
                    path, _ = robot.checked_joint_plan(arm, states[arm].joint_positions_rad,
                        np.asarray(a.terms['pregrasp_joint_positions_rad']),scene,c.target_id)
                except RuntimeError as error:
                    rejected.append(dict(ids=[c.candidate_id], reason=str(error)))
                    path = None
                paths[c.candidate_id] = path
        if any(paths[a.candidate.candidate_id] is None for a in pair):
            continue
        sampled = 0
        minimum = float('inf')
        for left, right in synchronized_knots(*(paths[a.candidate.candidate_id] for a in pair)):
            reason, clearance = check_spheres(robot.spheres(arms[0],left),robot.spheres(arms[1],right),[],None)
            minimum = min(minimum,clearance)
            sampled += 1
            if reason:break
        if reason:
            rejected.append(dict(ids=[a.candidate.candidate_id for a in pair], reason=reason))
            continue
        return dict(selection=[asdict(a.candidate) for a in pair],
                    slots=[a.terms.get('slot_xy') for a in pair],
                    pregrasp_knots=[np.asarray(paths[a.candidate.candidate_id]).tolist() for a in pair],
                    sample_count=sampled, min_clearance_m=minimum), rejected
    return None, rejected


def drive_empty_pregrasps(robot, paths, payload_guard=None):
    """Commit both empty-arm targets before each shared physics step."""
    arms=('panda_left','panda_right')
    previous=[robot.read_states()[a].joint_positions_rad.copy() for a in arms]
    if any(np.max(np.abs(q-np.asarray(path[0])))>.01 for q,path in zip(previous,paths)):
        raise RuntimeError('PAIR_START_STATE_CHANGED')
    start=np.array(previous)
    overlap=0
    ticks=0
    trace=[]
    def advance(desired):
        nonlocal previous,overlap,ticks
        # Also check tracking lag, not just the nominal synchronized paths.
        for left,right in synchronized_knots([previous[0],desired[0]],[previous[1],desired[1]]):
            if payload_guard:payload_guard(left,right)
            reason,_=check_spheres(robot.spheres(arms[0],left),robot.spheres(arms[1],right),[],None)
            if reason:raise RuntimeError('PAIR_COMMAND_'+reason)
        for a,q in zip(arms,desired):
            robot.targets[a][:7]=q
        robot.tick()
        states=robot.read_states()
        measured=np.array([states[a].joint_positions_rad for a in arms])
        if not np.isfinite(measured).all() or np.max(np.abs(measured-desired))>.04:
            raise RuntimeError('PAIR_JOINT_TRACKING_ERROR')
        reason,_=check_spheres(robot.spheres(arms[0],measured[0]),robot.spheres(arms[1],measured[1]),[],None)
        if reason:raise RuntimeError('PAIR_MEASURED_'+reason)
        if payload_guard:payload_guard(*measured)
        speed=np.max(np.abs(measured-previous),axis=1)/robot.dt
        moving=speed>.03
        overlap+=int(moving.all())
        ticks+=1
        trace.append(dict(tick=ticks,measured_max_joint_speed_rad_s=speed.tolist()))
        previous=measured
    for pair in synchronized_knots(*paths,max_step=min(.025,robot.joint_speed*robot.dt)):
        desired=np.asarray(pair)
        advance(desired)
        # Let the physical drives catch up before advancing the common clock.
        for _ in range(math.ceil(.25/robot.dt)):
            if np.max(np.abs(previous-desired))<=.009:break
            advance(desired)
        else:raise RuntimeError('PAIR_JOINT_TRACKING_ERROR')
    for _ in range(math.ceil(.3/robot.dt)):
        advance(np.asarray([p[-1] for p in paths]))
    moved=np.max(np.abs(previous-start),axis=1)
    if not overlap or np.min(moved)<.04:raise RuntimeError('NO_MEASURED_PARALLEL_APPROACH')
    return dict(ticks=ticks,overlap_ticks=overlap,overlap_sim_s=overlap*robot.dt,
                arm_order=list(arms),joint_displacements_rad=moved.tolist(),trace=trace)


def check_pair(env, args, packet, scene, estimator):
    from workstation.diagnostics.graspgen_calibration import verify
    from workstation.models.graspgen_client import visual_clouds
    from workstation.models.graspgen_proxy import GraspGenProxy
    from workstation.runtime_paths import isaac_root
    from workstation.simulation.robot_driver import from_environment

    report = dict(success=False, scope='snapshot_and_synchronized_empty_pregrasp_only',
                  physical_execution=False, simultaneous_grasp_verified=False, model_requests=[])
    def fail(reason):
        report['failure_reason'] = reason
        env._write_json('graspgen_pair_check.json', report)
        raise RuntimeError(reason)
    targets = [o for o in scene.objects if o.posture=='UPRIGHT' and not o.failures and pose_is_observed(o,scene)]
    if len(targets)!=2 or len({o.track_id for o in targets})!=2:
        fail('PAIR_REQUIRES_TWO_OBSERVED_UPRIGHT_TRACKS')
    execute=args.mode in ('graspgen-pair-pregrasp','graspgen-pair-place')
    if not execute:env.timeline.pause()
    robot=from_environment(env)
    robot.set_observation(packet,scene,ScenePriors.from_config(env.c))
    slots = ConveyorSlots(env.c)
    slots.observe(robot.observed_workspace)
    report['slots_initial'] = slots.report()
    available = [xy for xy in slots.positions if slots.states[xy] == 'free']
    if len(available) < 2:fail('TWO_OBSERVED_FREE_SLOTS_REQUIRED')
    calibration=verify(env,robot,isaac_root(),args.graspgen_assets)
    proxy=GraspGenProxy(args.graspgen_proxy)
    states=robot.read_states()
    candidates={}
    seq={n:f.sequence_id for n,f in packet.cameras.items() if f}
    rows=[]
    report['slot_assessments']=[]
    for attempt in range(1,4):
        for obj in targets:
            result=proxy.infer(visual_clouds(packet,scene,obj.track_id),args.num_grasps)
            proposals=candidates_from_result(result,obj,states,calibration,seq)
            candidates[obj.track_id]=[
                replace(c,rejection_reasons=c.rejection_reasons+('UNSUPPORTED_UPRIGHT_APPROACH',))
                if -c.approach_direction[2] < math.cos(math.radians(10)) else c for c in proposals]
            report['model_requests'].append(dict(target_id=obj.track_id,request_id=result['request_id'],
                                                 raw_grasps=len(result['scores'])))
        for slot in available:
            scored=assess(scene,states,lambda c,o,s:robot.feasibility(c,o,s,slot),robot.home,
                          candidate_provider=lambda o,_:candidates.get(o.track_id,[]))
            for row in scored:
                row.terms['slot_xy']=slot
            rows.extend(scored)
            for arm in states:
                assigned=[r for r in scored if r.candidate.arm==arm]
                report['slot_assessments'].append(dict(attempt=attempt,slot_xy=slot,arm=arm,
                    feasible=sum(r.feasible for r in assigned),
                    reasons=dict(Counter(reason for r in assigned for reason in r.reasons)),
                    phases=dict(Counter(r.terms['rejection_phase'] for r in assigned if r.terms.get('rejection_phase')))))
        report['feasible_per_arm']={a:len({row.candidate.candidate_id for row in rows if row.feasible and row.candidate.arm==a}) for a in states}
        pair,rejections=paired_pregrasp(robot,rows,states,scene)
        report.update(pair=pair, rejected_pairs=rejections)
        if pair is not None:break
    if pair is None:fail('NO_FEASIBLE_DISTINCT_GRASPGEN_PAIR')
    assigned_slots={c['arm']:xy for c,xy in zip(pair['selection'],pair['slots'])}
    slots.reserve_pair(assigned_slots)
    report['slots_reserved']=slots.report()
    if execute:
        try:
            robot.hold(.025)
            fresh=env.get_observation_packet(refresh=True)
            updated=estimator.estimate(fresh)
            robot.set_observation(fresh,updated,ScenePriors.from_config(env.c))
            selected={c['candidate_id'] for c in pair['selection']}
            seq={n:f.sequence_id for n,f in fresh.cameras.items() if f}
            new={}
            for a in rows:
                c=a.candidate
                if c.candidate_id not in selected or a.terms['slot_xy']!=assigned_slots[c.arm]:continue
                matches=[o for o in updated.objects if o.track_id==c.target_id]
                if len(matches)!=1:raise RuntimeError('PAIR_TARGET_LOST')
                new[c.target_id]=[revalidate_candidate(c,matches[0],updated,seq)]
            slots.observe(robot.observed_workspace)
            if any(slots.states[xy]!='reserved' for xy in assigned_slots.values()):
                raise RuntimeError('RESERVED_SLOT_CHANGED')
            states=robot.read_states()
            checked=assess(updated,states,lambda c,o,s:robot.feasibility(c,o,s,assigned_slots[c.arm]),robot.home,
                           candidate_provider=lambda o,_:new.get(o.track_id,[]))
            for row in checked:
                row.terms['slot_xy']=assigned_slots[row.candidate.arm]
            current,_=paired_pregrasp(robot,checked,states,updated)
            if current is None:raise RuntimeError('PAIR_CHANGED_BEFORE_APPROACH')
            report.update(pair=current,scope='physical_empty_arm_pregrasp_only',physical_execution=True)
            env._write_json('graspgen_pair_check.json',report)
            action_start=time.monotonic()
            report['motion']=drive_empty_pregrasps(robot,current['pregrasp_knots'])
            if args.mode=='graspgen-pair-place':
                from workstation.diagnostics.graspgen_pair_place import place_pair
                report['scope']='two_cartons_same_physics_scene'
                try:
                    place_pair(env,robot,estimator,updated,current,slots,report)
                finally:
                    report['action_wall_s']=time.monotonic()-action_start
                    env._write_json('graspgen_pair_check.json',report)
        except RuntimeError as error:
            fail(str(error))
    report['success']=True
    env._write_json('graspgen_pair_check.json',report)
