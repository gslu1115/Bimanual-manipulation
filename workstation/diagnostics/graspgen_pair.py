"""GraspGen proposals for two distinct visual tracks in one frozen Isaac scene."""
from dataclasses import asdict, replace
from collections import Counter
from itertools import product
import math
import time

import numpy as np

from workstation.perception.scene_estimate import pose_is_observed, ScenePriors
from workstation.planning.collision import check_spheres
from workstation.planning.graspgen_candidates import candidates_from_result, revalidate_candidate, refine_pinch_contact
from workstation.planning.selection import assess
from workstation.planning.task_targets import ConveyorSlots


def renew_view_candidate(candidate,scene,seq):
    """Reconnect a model pose to a unique fresh observation after wrist motion."""
    source=candidate.metadata['source_visual_object']
    matches=[o for o in scene.objects if o.track_id==candidate.target_id]
    if not matches:
        matches=[o for o in scene.objects if
            np.linalg.norm(np.asarray(o.position_m)-source['position_m'])<=.002
            and np.allclose(o.size_m,source['size_m'],atol=1e-6,rtol=0)]
    if len(matches)!=1:raise RuntimeError('PAIR_TARGET_LOST')
    return revalidate_candidate(replace(candidate,target_id=matches[0].track_id),matches[0],scene,seq)


def recover_returned_carton(candidate,packet,scene,priors):
    """Resolve a clipped upright footprint after an inspection wrist returns home."""
    from workstation.perception.cuboid_pose import fit_cuboid_pose, measured_cuboid_supported
    from workstation.observations.camera_geometry import quaternion_from_matrix
    src=candidate.metadata['source_visual_object']
    near=[o for o in scene.objects if np.linalg.norm(np.subtract(o.position_m,src['position_m']))<.07]
    if len(near)!=1:return None
    obj,=near
    if (obj.posture!='UPRIGHT' or obj.failures or obj.quality.get('packaging_conflict') or
            obj.velocity_m_s is None or np.linalg.norm(obj.velocity_m_s)>.015):return None
    # Both source and current target rest on the calibrated table. A smaller
    # footprint cannot explain a top whose measured height matches another size.
    observed_height=obj.quality.get('observed_top_z_m',float('nan'))-priors.support_z_m
    sizes=[s for s in priors.carton_size_catalog_m if abs(s[2]-observed_height)<=.002]
    if len(sizes)!=1 or not np.allclose(sizes[0],src['size_m'],rtol=0,atol=1e-6):return None
    m=fit_cuboid_pose(packet,scene,src['position_m'],src['body_rotation'],sizes[0],priors,measure_side_interiors=True)
    if (not measured_cuboid_supported(m,packet.observation_time_s) or set(m['mask_refs'])!=set(obj.mask_refs)
            or len({ref.split('/')[0] for ref in m['mask_refs']})<2):return None
    r=np.asarray(m['rotation'])
    if r[2,2]<math.cos(math.radians(3)):return None
    updated=replace(obj,position_m=tuple(m['position_m']),size_m=tuple(sizes[0]),
        axis_yaw_rad=math.atan2(r[1,0],r[0,0]),measured_cuboid=m,visibility='OBSERVED_CUBOID',
        hypotheses=tuple(replace(h,orientation_wxyz=tuple(quaternion_from_matrix(r))) for h in obj.hypotheses if h.label=='UPRIGHT'),
        uncertainty_m=m['uncertainty_m'])
    if not updated.hypotheses:return None
    return replace(scene,objects=tuple(updated if o is obj else o for o in scene.objects))


def source_cartons(scene, table):
    """Select visible cartons on the pickup table; keep belt objects in the obstacle map."""
    centre=np.asarray(table['center_xy'])
    half=np.asarray(table['size'][:2])/2
    return [o for o in scene.objects if o.posture=='UPRIGHT' and not o.failures
            and np.all(np.abs(np.asarray(o.position_m[:2])-centre)<=half)
            and pose_is_observed(o,scene)]


def preferred_assignments(targets, bases):
    """Assign both cartons to distinct arms with the shortest total XY reach."""
    left,right=sorted(bases)
    def distance(arm,obj):
        return np.linalg.norm(np.asarray(bases[arm]['base_xyz'])[:2]-np.asarray(obj.position_m)[:2])
    if len(targets)>2:return {o.track_id:min(bases,key=lambda arm:distance(arm,o)) for o in targets}
    a,b=targets
    if distance(left,b)+distance(right,a)<distance(left,a)+distance(right,b):
        a,b=b,a
    return {a.track_id:left,b.track_id:right}


def balance_visible_surfaces(clouds,obj):
    """Reduce top-view pixel density while retaining actual measured side surfaces."""
    from workstation.planning.graspgen_candidates import upright_body_rotation
    points=clouds.target_m
    local=(points-np.asarray(obj.position_m))@upright_body_rotation(obj)
    _,indices=np.unique(np.floor(local/.0015).astype(int),axis=0,return_index=True)
    indices=np.sort(indices)
    residual=np.abs(np.abs(local[indices])-np.asarray(obj.size_m)/2)
    closest=np.argmin(residual,axis=1)
    side=indices[closest!=2]
    top=indices[closest==2]
    if len(side)<40 or len(indices)<200:return clouds
    budget=min(len(top),max(200-len(side),2*len(side)))
    selected=np.sort(np.r_[side,top[np.linspace(0,len(top)-1,budget,dtype=int)]])
    return replace(clouds,target_m=points[selected])


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


def paired_pregrasp(robot, rows, states, scene,selection_policy=None,complete_assessment=None,*,prefix_assessment=None,pick_cache=None):
    arms = ('panda_left', 'panda_right')
    feasible = [[a for a in rows if a.feasible and a.candidate.source == 'graspgen'
                 and a.candidate.arm == arm and (a.terms.get('full_pose_checked') or
                 (complete_assessment is not None and a.terms.get('ranking_bound')))] for arm in arms]
    pairs = [(left, right) for left, right in product(*feasible)
             if left.candidate.target_id != right.candidate.target_id
             and ('slot_xy' not in left.terms or left.terms['slot_xy'] != right.terms['slot_xy'])]
    def slot_cost(pair):
        # Prefer independent lanes when free. The centre remains available
        # to either arm when a side lane is occupied or unreachable.
        return sum(abs(row.candidate.metadata.get('source_visual_object',{}).get(
                       'position_m',row.candidate.position_m)[0]-row.terms['slot_xy'][0])
                   for row in pair if row.terms.get('slot_xy') is not None)
    def rank(pair):
        return ((selection_policy.priority(pair) if selection_policy else ())+
                (slot_cost(pair),-(pair[0].score+pair[1].score)))
    pairs.sort(key=rank)
    if prefix_assessment is not None:
        from workstation.planning.plan_reuse import refine_pick_pairs
        ordered=refine_pick_pairs(pairs,rank,prefix_assessment,complete_assessment)
    elif complete_assessment is not None:
        from workstation.planning.plan_reuse import complete_ranked_pairs
        ordered=complete_ranked_pairs(pairs,rank,complete_assessment)
    else:ordered=pairs
    paths = {}
    rejected = []
    for pair in ordered:
        for arm, a in zip(arms, pair):
            c = a.candidate
            if c.candidate_id not in paths:
                try:
                    obj=next((o for o in scene.objects if o.track_id==c.target_id),None) if pick_cache else None
                    prior=(pick_cache.pregrasps.get(pick_cache.key(robot,c,obj,scene,None,True))
                           if obj is not None else None)
                    if prior is not None and np.array_equal(prior[0],a.terms['pregrasp_joint_positions_rad']):
                        path=prior[3]
                        pick_cache.path_hits+=1
                    else:
                        path,_=robot.checked_joint_plan(arm,states[arm].joint_positions_rad,
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
        result=dict(selection=[asdict(a.candidate) for a in pair],
                    slots=[a.terms.get('slot_xy') for a in pair],
                    pregrasp_knots=[np.asarray(paths[a.candidate.candidate_id]).tolist() for a in pair],
                    sample_count=sampled, min_clearance_m=minimum)
        if selection_policy:
            result['joint_choice']=dict(source_keys=[selection_policy.key(a.candidate) for a in pair],
                history_penalty=selection_policy.priority(pair)[0],visibility=selection_policy.visibility(pair),
                feasible_pair_count=len(pairs),
                pair_count_scope='optimistic combinations, fully validated in rank order' if complete_assessment else 'fully assessed combinations',
                eligible_source_count=len({r.candidate.target_id for r in rows if r.feasible}))
        return result, rejected
    return None, rejected


def drive_empty_pregrasps(robot, paths, payload_guard=None, following_band=.025,speed_fraction=1.,
                          require_overlap=True,loaded_transfer=False,progress_check=None,
                          trajectory_kind='trajectory',settle_seconds=.3,collision_obstacles=None):
    """Commit both empty-arm targets before each shared physics step."""
    arms=('panda_left','panda_right')
    states=robot.read_states()
    previous=[states[a].joint_positions_rad.copy() for a in arms]
    if any(np.max(np.abs(q-np.asarray(path[0])))>.01 for q,path in zip(previous,paths)):
        raise RuntimeError('PAIR_START_STATE_CHANGED')
    start=np.array(previous)
    commanded=start.copy()
    overlap=0
    ticks=0
    trace=[]
    tick_counts=dict(trajectory=0,tracking_wait=0,settling=0)
    last_geometry=None
    geometry_checks=dict(computed=0,reused=0)
    def arm_clearance(left,right):
        nonlocal last_geometry
        pair=(robot.spheres(arms[0],left),robot.spheres(arms[1],right))
        if last_geometry is not None and all(a is b for a,b in zip(pair,last_geometry)):
            geometry_checks['reused']+=1
            return None
        if collision_obstacles is None:
            reason,_=check_spheres(*pair,[],None)
        else:
            # One predicate per arm: left also checks mutual separation.
            # The same frozen obstacles apply to commanded and measured poses.
            reason,_=check_spheres(*pair,collision_obstacles,arms[0]+'_mount')
            if reason is None:
                reason,_=check_spheres(pair[1],[],collision_obstacles,arms[1]+'_mount')
        geometry_checks['computed']+=1
        last_geometry=pair if reason is None else None
        return reason
    def advance(desired,kind='trajectory'):
        nonlocal previous,overlap,ticks,commanded
        # Also check tracking lag, not just the nominal synchronized paths.
        for left,right in synchronized_knots([previous[0],desired[0]],[previous[1],desired[1]]):
            if payload_guard:payload_guard(left,right)
            reason=arm_clearance(left,right)
            if reason:raise RuntimeError('PAIR_COMMAND_'+reason)
        for i,(a,q) in enumerate(zip(arms,desired)):
            robot.targets[a][:7]=q
            if hasattr(robot,'command_velocities'):
                feedforward=(q-commanded[i])/robot.dt
                # Follow the measured arm through the waypoint, including a
                # bounded correction while its partner waits on the same tick.
                lag=commanded[i]-previous[i]
                correction=(np.clip(10.*lag,-.28,.28) if loaded_transfer
                            else np.clip(6.*lag,-.15,.15))
                robot.command_velocities(a,np.r_[np.clip(feedforward+correction,
                    -robot.joint_speed,robot.joint_speed),0.,0.])
        commanded=desired.copy()
        previous_kind=getattr(robot,'_tick_kind','trajectory')
        robot._tick_kind=trajectory_kind if kind=='trajectory' else kind
        try:robot.tick()
        finally:robot._tick_kind=previous_kind
        tick_counts[kind]+=1
        states=robot.read_states()
        measured=np.array([states[a].joint_positions_rad for a in arms])
        if not np.isfinite(measured).all():
            raise RuntimeError('PAIR_JOINT_TRACKING_ERROR: nonfinite encoders')
        errors=np.abs(measured-desired)
        if np.max(errors)>.04:
            i,j=np.unravel_index(np.argmax(errors),errors.shape)
            raise RuntimeError(f'PAIR_JOINT_TRACKING_ERROR: {arms[i]} joint {j+1} '
                               f'error {errors[i,j]:.4f} rad at tick {ticks}')
        reason=arm_clearance(*measured)
        if reason:raise RuntimeError('PAIR_MEASURED_'+reason)
        if payload_guard:payload_guard(*measured)
        speed=np.max(np.abs(measured-previous),axis=1)/robot.dt
        moving=speed>.03
        overlap+=int(moving.all())
        ticks+=1
        trace.append(dict(tick=ticks,measured_max_joint_speed_rad_s=speed.tolist()))
        previous=measured
        if progress_check is not None:
            progress_check(ticks*robot.dt)
    for pair in synchronized_knots(*paths,max_step=min(.025,speed_fraction*robot.joint_speed*robot.dt)):
        desired=np.asarray(pair)
        advance(desired)
        # Let the physical drives catch up before advancing the common clock.
        for _ in range(math.ceil((.45 if loaded_transfer else .25)/robot.dt)):
            if np.max(np.abs(previous-desired))<=following_band:break
            advance(desired,'tracking_wait')
        else:
            errors=np.max(np.abs(previous-desired),axis=1)
            raise RuntimeError('PAIR_JOINT_TRACKING_ERROR: '+str(dict(zip(arms,errors.tolist()))))
    for _ in range(math.ceil(settle_seconds/robot.dt)):
        advance(np.asarray([p[-1] for p in paths]),'settling')
    moved=np.max(np.abs(previous-start),axis=1)
    if require_overlap and (not overlap or np.min(moved)<.04):raise RuntimeError('NO_MEASURED_PARALLEL_APPROACH')
    return dict(ticks=ticks,geometry_checks=geometry_checks,overlap_ticks=overlap,overlap_sim_s=overlap*robot.dt,
                arm_order=list(arms),joint_displacements_rad=moved.tolist(),trace=trace,
                tick_counts=tick_counts,sim_seconds={k:v*robot.dt for k,v in tick_counts.items()})


class PairAttemptError(RuntimeError):
    def __init__(self,reason,report):
        super().__init__(reason);self.report=report


def approach_alternatives(rows,assigned_slots,completed_rows=None):
    """Retain pending backups for on-demand validation by the substitution loop."""
    completed_rows=completed_rows or {}
    rows=[completed_rows.get((r.candidate.candidate_id,tuple(r.terms['slot_xy'])),r) for r in rows]
    alternatives={arm:[] for arm in assigned_slots};seen=set()
    for row in sorted(rows,key=lambda item:-item.score):
        c=row.candidate
        if row.feasible and row.terms['slot_xy']==assigned_slots[c.arm] and c.candidate_id not in seen:
            alternatives[c.arm].append(c);seen.add(c.candidate_id)
    return alternatives


def check_pair(env, args, packet, scene, estimator, *, robot=None, target_selector=None,selection_policy=None):
    from workstation.diagnostics.graspgen_calibration import verify
    from workstation.models.graspgen_client import visual_clouds, infer_rotated_view
    from workstation.planning.selection import visual_pinch_balance, upright_contact_cost, distinct_pinches
    from workstation.models.graspgen_proxy import GraspGenProxy
    from workstation.runtime_paths import isaac_root
    from workstation.simulation.robot_driver import from_environment

    report = dict(success=False, scope='snapshot_and_synchronized_empty_pregrasp_only',
                  physical_execution=False, simultaneous_grasp_verified=False, model_requests=[])
    timing=getattr(robot,'wall_timing',None)
    wait_start=time.perf_counter()
    stage_start=timing.stage_snapshot() if timing is not None else {}
    preparation_start=timing.snapshot() if timing is not None else None
    preparation_validation_start=dict(getattr(robot,'validation_reuse',{}))
    def planning_counts():
        return {key:getattr(robot,key,0) for key in
                ('full_forecasts_started','full_forecasts_passed','cartesian_reuses')}
    def count_delta(before):
        return {key:value-before[key] for key,value in planning_counts().items()}
    planning_start=planning_counts()
    def fail(reason):
        report['failure_reason'] = reason
        env._write_json('graspgen_pair_check.json', report)
        raise PairAttemptError(reason,report)
    select_targets=target_selector or (lambda current:source_cartons(current,env.c['table']))
    targets=select_targets(scene)
    if ((len(targets)<2 if selection_policy else len(targets)!=2) or len({o.track_id for o in targets})!=len(targets)):
        fail('PAIR_REQUIRES_TWO_OBSERVED_UPRIGHT_TRACKS')
    execute=args.mode in ('graspgen-pair-pregrasp','graspgen-pair-place')
    if not execute:env.timeline.pause()
    armature_floor=.01 if execute else None
    if robot is None:robot=from_environment(env,paired_armature_floor=armature_floor)
    report['shoulder_drive_armature_floor_kg_m2']=armature_floor
    estimator.robot_point_filter=robot.observed_robot_points
    estimator.robot_filter_factory=robot.prepare_observed_robot_filter
    robot.set_observation(packet,scene,ScenePriors.from_config(env.c))
    if selection_policy:selection_policy.prepare(robot,packet,scene)
    diversity_start=len(selection_policy.diversity) if selection_policy else 0
    slots = ConveyorSlots(env.c)
    destinations=[slots.positions[i] for i in env.c.get('pair_slot_indices',range(len(slots.positions)))]
    report['eligible_destinations']=destinations
    slots.observe(robot.observed_workspace)
    report['slots_initial'] = slots.report()
    from workstation.diagnostics.graspgen_pair_place import stationary_slot_tracks
    parked=stationary_slot_tracks(robot.observed_workspace,scene,slots.positions)
    report['parked_before_pick']=[dict(xy=list(k),track_id=o.track_id,position_m=list(o.position_m),
        axis_yaw_rad=o.axis_yaw_rad,size_m=list(o.size_m)) for k,o in parked.items()]
    available = [xy for xy in destinations if slots.states[xy] == 'free']
    if len(available) < 2:fail('TWO_OBSERVED_FREE_SLOTS_REQUIRED')
    calibration=verify(env,robot,isaac_root(),args.graspgen_assets)
    proxy=GraspGenProxy(args.graspgen_proxy)
    states=robot.read_states()
    candidates={}
    seq={n:f.sequence_id for n,f in packet.cameras.items() if f}
    rows=[]
    inspection_cloud=None
    report['slot_assessments']=[]
    pick_failures={}
    from workstation.planning.plan_reuse import CandidatePrefixCache
    pick_prefixes=CandidatePrefixCache()
    completed_rows={}
    prefix_rows={}
    def prefix_assessment(row):
        c=row.candidate
        slot=row.terms['slot_xy']
        key=c.candidate_id,tuple(slot)
        if key not in prefix_rows:
            checked=assess(scene,states,lambda c,o,s:check_destination(c,o,s,slot,False),robot.home,
                           candidate_provider=lambda o,_:[c] if o.track_id==c.target_id else [])
            result=checked[0]
            result.terms.update(slot_xy=slot,source_order=row.terms['source_order'])
            prefix_rows[key]=result
        return prefix_rows[key]
    def complete_assessment(row):
        c=row.candidate;slot=row.terms['slot_xy'];key=(c.candidate_id,tuple(slot))
        if key not in completed_rows:
            checked=assess(scene,states,
                lambda c,o,s:robot.feasibility(c,o,s,slot,prefix_cache=pick_prefixes),robot.home,
                candidate_provider=lambda o,_:[c] if o.track_id==c.target_id else [])
            result=checked[0];result.terms['slot_xy']=slot
            completed_rows[key]=result
        return completed_rows[key]
    def check_destination(c,o,s,slot,rank_only=True):
        # Pick IK, empty approach and finger closure do not depend on the
        # downstream slot. Reuse their failure only for this frozen snapshot.
        if c.candidate_id in pick_failures:return pick_failures[c.candidate_id]
        result=robot.feasibility(c,o,s,slot,prefix_cache=pick_prefixes,prefix_only=selection_policy is not None,rank_only=selection_policy is not None and rank_only)
        if result.get('reasons') and result.get('rejection_phase') in ('PREGRASP','APPROACH','CLOSE'):
            pick_failures[c.candidate_id]=result
        return result
    for attempt in range(1,3 if selection_policy else 8):
        retained={}
        preferred=preferred_assignments(targets,robot.base_config)
        feasible={(r.candidate.target_id,r.candidate.arm) for r in rows if r.feasible}
        missing=[o for o in targets if (o.track_id,preferred[o.track_id]) not in feasible]
        if execute and attempt==5 and len(missing)==1:
            target=missing[0]
            centre=np.asarray(target.position_m)
            arm=preferred[target.track_id]
            camera='left_wrist_camera' if arm=='panda_left' else 'right_wrist_camera'
            observation=dict(arm=arm,target_id=target.track_id,executed=False,trials=[])
            report['extra_view']=observation
            before_view=visual_clouds(packet,scene,target.track_id)
            for side in (1.,-1.):
                lateral=np.array([-math.sin(target.axis_yaw_rad),math.cos(target.axis_yaw_rad),0.])
                goal=centre+side*.18*lateral+[0.,0.,.19]
                try:
                    view=robot.execute_camera_view(arm,packet.cameras[camera],goal,centre,scene,target.track_id)
                except RuntimeError as error:
                    observation['trials'].append(str(error))
                    if robot.motion_started:fail('INSPECTION_MOTION:'+str(error))
                    continue
                observation.update(executed=True,view=view)
                packet=env.get_observation_packet(refresh=True)
                scene=estimator.estimate(packet)
                robot.hold(.10)
                packet=env.get_observation_packet(refresh=True)
                scene=estimator.estimate(packet)
                robot.set_observation(packet,scene,ScenePriors.from_config(env.c))
                states=robot.read_states()
                slots.observe(robot.observed_workspace)
                available=[xy for xy in destinations if slots.states[xy]=='free']
                targets=select_targets(scene)
                if len(targets)!=2 or len(available)<2:fail('INSPECTION_SCENE_NOT_READY')
                matches=[o for o in targets if np.linalg.norm(np.asarray(o.position_m)-centre)<=.002
                         and np.allclose(o.size_m,target.size_m,atol=1e-6,rtol=0)
                         and abs((o.axis_yaw_rad-target.axis_yaw_rad+math.pi/2)%math.pi-math.pi/2)<=math.radians(3)
                         and o.velocity_m_s is not None and np.linalg.norm(o.velocity_m_s)<=.015]
                if len(matches)==1:
                    inspection_cloud=(matches[0].track_id,before_view)
                old={row.candidate.candidate_id:row.candidate for row in rows if row.feasible}
                rows.clear()
                pick_failures.clear()
                completed_rows.clear()
                seq={n:f.sequence_id for n,f in packet.cameras.items() if f}
                for candidate in old.values():
                    try:renewed=renew_view_candidate(candidate,scene,seq)
                    except RuntimeError:continue
                    retained.setdefault(renewed.target_id,[]).append(renewed)
                observation['revalidated_prior_proposals']=sum(map(len,retained.values()))
                missing=[o for o in targets if o.track_id not in retained]
                break
        requested=missing or targets
        candidates=retained
        for obj in requested:
            # A distant camera and a near wrist see different faces. On retry
            # draw from different actual views of the missing carton, keeping
            # the feasible results for every previously sampled carton.
            refs=obj.mask_refs
            if attempt==2:
                chosen=[ref for ref in refs if ref.startswith('scene_camera/')]
                if chosen and sum(np.count_nonzero(scene.masks[ref]) for ref in chosen)>=200:
                    refs=chosen
            source_scene=replace(scene,objects=tuple(replace(o,mask_refs=tuple(refs))
                if o.track_id==obj.track_id else o for o in scene.objects))
            yaw=(0.,-obj.axis_yaw_rad,math.pi/2-obj.axis_yaw_rad,math.pi/4,-math.pi/4,
                 math.pi-obj.axis_yaw_rad,-math.pi/2-obj.axis_yaw_rad)[attempt-1]
            clouds=visual_clouds(packet,source_scene,obj.track_id)
            if inspection_cloud and inspection_cloud[0]==obj.track_id:
                past=inspection_cloud[1]
                # Both captures were deprojected with their own image-time FK.
                # Reuse only target surfaces for proposal generation. The
                # newest RGB-D packet alone supplies the collision scene.
                clouds=replace(clouds,target_m=np.concatenate((clouds.target_m,past.target_m)))
                observation['fused_observation_times_s']=[past.time_s,clouds.time_s]
            if attempt>=3:clouds=balance_visible_surfaces(clouds,obj)
            proposals=[]
            for sample in range(3 if attempt<=2 else 9):
                basis=yaw+sample*math.pi*2/3
                pitch=(0.,math.pi/3,-math.pi/3)[sample//3]
                result=infer_rotated_view(proxy,clouds,args.num_grasps,basis,pitch)
                proposals.extend(candidates_from_result(result,obj,states,calibration,seq))
                report['model_requests'].append(dict(target_id=obj.track_id,request_id=result['request_id'],
                    attempt=attempt,mask_refs=list(refs),basis_yaw_rad=basis,basis_pitch_rad=pitch,
                    observed_input_points=len(clouds.target_m),raw_grasps=len(result['scores'])))
            checked=[]
            for c in proposals:
                c=refine_pinch_contact(c,obj)
                reasons=c.rejection_reasons
                if -c.approach_direction[2]<math.cos(math.radians(10)):
                    reasons+=('UNSUPPORTED_UPRIGHT_APPROACH',)
                if not reasons:
                    balance=visual_pinch_balance(c,obj)
                    if balance['offset_m']>.5*balance['half_extent_m']:
                        reasons+=('ECCENTRIC_PAIRED_LIFT',)
                checked.append(replace(c,rejection_reasons=reasons))
            # Each full-pose check forecasts approach, lift and placement.
            # Try balanced pinches first on the nearer arm. Keep the other
            # arm eligible in later rounds when the local lane has failed.
            allowed=[c for c in checked if not c.rejection_reasons]
            report.setdefault('proposal_filters',[]).append(dict(attempt=attempt,target_id=obj.track_id,
                candidate_count=len(checked),eligible=len(allowed),
                reasons=dict(Counter(reason for c in checked for reason in c.rejection_reasons))))
            def pinch_cost(c):
                opening=c.metadata['opening']
                offset=visual_pinch_balance(c,obj)['penalty']
                depth=opening.get('estimated_tip_engagement_m',0.)/obj.size_m[2]
                return offset+abs(depth-.45)+abs(opening['centre_offset_m'])/.02+upright_contact_cost(c,obj)
            for arm in states:
                if selection_policy is None and attempt<=4 and arm!=preferred.get(obj.track_id):continue
                ordered=sorted((c for c in allowed if c.arm==arm),key=pinch_cost)
                chosen=(selection_policy.diverse(ordered,obj,4) if selection_policy else distinct_pinches(ordered,4))
                candidates.setdefault(obj.track_id,[]).extend(chosen)
        order={c.candidate_id:i for group in candidates.values() for i,c in enumerate(group)}
        for slot in available:
            scored=assess(scene,states,lambda c,o,s:check_destination(c,o,s,slot),robot.home,
                          candidate_provider=lambda o,_:candidates.get(o.track_id,[]))
            for row in scored:
                # Preserve the eager sort's append order across resampling rounds.
                row.terms.update(slot_xy=slot,source_order=(len(rows),order[row.candidate.candidate_id]))
            rows.extend(scored)
            for arm in states:
                assigned=[r for r in scored if r.candidate.arm==arm]
                report['slot_assessments'].append(dict(attempt=attempt,slot_xy=slot,arm=arm,
                    scope='pregrasp ranking bound; pick and destination pending' if selection_policy else 'whole path',
                    feasible=sum(r.feasible for r in assigned),
                    feasible_by_target=dict(Counter(r.candidate.target_id for r in assigned if r.feasible)),
                    reasons=dict(Counter(reason for r in assigned for reason in r.reasons)),
                    phases=dict(Counter(r.terms['rejection_phase'] for r in assigned if r.terms.get('rejection_phase')))))
        report['candidate_prefix_reuse']=pick_prefixes.report()
        report['feasible_per_arm']={a:len({row.candidate.candidate_id for row in rows if row.feasible and row.candidate.arm==a}) for a in states}
        report['feasible_per_arm_scope']='pregrasp passed; remaining stages pending ranked validation' if selection_policy else 'whole path'
        pair,rejections=paired_pregrasp(robot,rows,states,scene,selection_policy,
                                      complete_assessment if selection_policy else None,
                                      prefix_assessment=prefix_assessment if selection_policy else None,pick_cache=pick_prefixes)
        report['candidate_prefix_reuse']=pick_prefixes.report()
        report['destination_evaluation']=dict(mode='pregrasp upper bound; ranked pick and destination refinement' if selection_policy else 'exhaustive',
            ranked_rows=len(rows),prefix_rows=len(prefix_rows) if selection_policy else len(rows),completed_rows=len(completed_rows),
            feasible_completed_rows=sum(r.feasible for r in completed_rows.values()),
            skipped_destination_rows=sum(r.feasible and (r.candidate.candidate_id,tuple(r.terms['slot_xy'])) not in completed_rows for r in rows),
            failures=[dict(candidate_id=r.candidate.candidate_id,slot_xy=r.terms['slot_xy'],reasons=r.reasons)
                      for r in completed_rows.values() if not r.feasible])
        report.update(pair=pair, rejected_pairs=rejections)
        if selection_policy:report['grasp_diversity']=selection_policy.diversity[diversity_start:]
        if pair is not None:break
        if selection_policy and attempt==1:
            # Retry requests depend on actual prefix feasibility on each arm.
            rows=[prefix_assessment(row) if row.terms.get('ranking_bound') else row for row in rows]
    if pair is None:fail('NO_FEASIBLE_DISTINCT_GRASPGEN_PAIR')
    assigned_slots={c['arm']:xy for c,xy in zip(pair['selection'],pair['slots'])}
    slots.reserve_pair(assigned_slots)
    report['slots_reserved']=slots.report()
    if execute:
        action_start=None
        try:
            returned=report.get('extra_view',{}).get('executed',False)
            if returned:
                observer=report['extra_view']['arm']
                # Keep the grasps inferred from the side view, but leave the
                # folded inspection posture through a checked empty-hand path.
                robot.execute_joint(observer,robot.home,scene,None)
                fresh=env.get_observation_packet(refresh=True)
                estimator.estimate(fresh)
                robot.hold(.10)
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
                try:
                    renewed=renew_view_candidate(c,updated,seq)
                except RuntimeError as error:
                    recovered=None
                    if returned and str(error)=='PAIR_TARGET_LOST':
                        recovered=recover_returned_carton(c,fresh,updated,estimator.priors)
                    if recovered is not None:
                        renewed=renew_view_candidate(c,recovered,seq)
                        updated=recovered
                        robot.set_observation(fresh,updated,estimator.priors)
                        report.setdefault('return_size_refits',[]).append(dict(arm=c.arm,
                            source_track=c.target_id,current_track=renewed.target_id,time_s=fresh.observation_time_s))
                    else:
                        from workstation.diagnostics.vision_check import save_observation
                        save_observation(env.output,'pair_target_revalidation',fresh,updated)
                        report['target_revalidation_failure']=dict(arm=c.arm,target_id=c.target_id,
                            candidate_id=c.candidate_id,source=c.metadata['source_visual_object'],
                            returned_from_extra_view=returned,reason=str(error))
                        raise
                if renewed.target_id!=c.target_id:
                    report.setdefault('reidentified_after_view',{})[c.target_id]=renewed.target_id
                new[renewed.target_id]=[renewed]
            slots.observe(robot.observed_workspace)
            if any(slots.states[xy]!='reserved' for xy in assigned_slots.values()):
                raise RuntimeError('RESERVED_SLOT_CHANGED')
            states=robot.read_states()
            fresh_prefixes=CandidatePrefixCache()
            checked=assess(updated,states,lambda c,o,s:robot.feasibility(c,o,s,assigned_slots[c.arm],prefix_cache=fresh_prefixes),robot.home,
                           candidate_provider=lambda o,_:new.get(o.track_id,[]))
            for row in checked:
                row.terms['slot_xy']=assigned_slots[row.candidate.arm]
            current,_=paired_pregrasp(robot,checked,states,updated,pick_cache=fresh_prefixes)
            if current is None:raise RuntimeError('PAIR_CHANGED_BEFORE_APPROACH')
            report.update(pair=current,scope='physical_empty_arm_pregrasp_only',physical_execution=True)
            if selection_policy:
                report['joint_choice']=pair['joint_choice']
            env._write_json('graspgen_pair_check.json',report)
            action_start=time.monotonic()
            if timing is not None:
                report['preparation_timing']=timing.delta(preparation_start)
                report['preparation_path_stage_detail']=timing.stage_delta(stage_start)
                report['pair_entry_to_first_motion_wall_s']=time.perf_counter()-wait_start
                report['ranked_pick_work']=dict(prefix_rows_checked=len(prefix_rows),full_slot_rows=len(completed_rows),
                    selected_rows_revalidated=len(checked),pregrasp_paths_reused=pick_prefixes.path_hits+fresh_prefixes.path_hits)
                report['preparation_planning_counts']=count_delta(planning_start)
                action_timing_start=timing.snapshot()
            validation_start=dict(getattr(robot,'validation_reuse',{}))
            action_planning_start=planning_counts()
            report['preparation_validation_reuse']={k:v-preparation_validation_start.get(k,0) for k,v in validation_start.items()}
            action_ticks_start=dict(getattr(robot,'tick_counts',{}))
            reader=getattr(robot,'encoder_snapshot_reader',None)
            encoder_start=(reader.reads,reader.reuses) if reader is not None else (0,0)
            report['motion']=drive_empty_pregrasps(robot,current['pregrasp_knots'],speed_fraction=.5)
            for selected,path in zip(current['selection'],current['pregrasp_knots']):
                arm=selected['arm']
                expected,orientation=robot.fk(arm,np.asarray(path[-1]))
                actual,measured=robot.fk(arm)
                position_error=float(np.linalg.norm(actual-expected))
                angle_error=float(np.arccos(np.clip((np.trace(measured.T@orientation)-1)/2,-1.,1.)))
                report['motion'].setdefault('endpoint_errors',{})[arm]=dict(position_m=position_error,angle_rad=angle_error)
                # This waypoint is in free air. The descent is replanned from
                # measured FK and separately checked to 5 mm at actual contact.
                if position_error>.008 or angle_error>math.radians(1.5):raise RuntimeError('PAIR_PREGRASP_GOAL_MISSED:'+arm)
            if args.mode=='graspgen-pair-place':
                from workstation.diagnostics.graspgen_pair_place import place_pair
                report['scope']='two_cartons_same_physics_scene'
                alternatives=approach_alternatives(rows,assigned_slots,{**prefix_rows,**completed_rows})
                def resample(arm,obj,view,image,previous,goal):
                    seq={n:f.sequence_id for n,f in image.cameras.items() if f}
                    observed=visual_clouds(image,view,obj.track_id)
                    options=[]
                    for yaw in (0.,-obj.axis_yaw_rad,math.pi/2-obj.axis_yaw_rad):
                        result=infer_rotated_view(proxy,observed,args.num_grasps,yaw)
                        report['model_requests'].append(dict(target_id=obj.track_id,request_id=result['request_id'],
                            purpose='one_arm_regrasp',raw_grasps=len(result['scores'])))
                        for raw in candidates_from_result(result,obj,(arm,),calibration,seq):
                            c=refine_pinch_contact(raw,obj)
                            if c.rejection_reasons or -c.approach_direction[2]<math.cos(math.radians(10)):continue
                            if (np.linalg.norm(np.subtract(c.position_m,previous['position_m']))<.004 and
                                abs(np.dot(c.orientation_wxyz,previous['orientation_wxyz']))>math.cos(math.radians(4)/2)):continue
                            options.append(c)
                    options.sort(key=lambda c:visual_pinch_balance(c,obj)['penalty']+upright_contact_cost(c,obj))
                    verified=assess(view,robot.read_states(),lambda c,o,s:robot.feasibility(c,o,s,goal),robot.home,
                        candidate_provider=lambda o,_:distinct_pinches(options,6) if o.track_id==obj.track_id else [])
                    best=next((r.candidate for r in verified if r.feasible),None)
                    if best is None:raise RuntimeError('NO_FEASIBLE_REGRASP')
                    return best
                old_keepout=getattr(robot,'stationary_payload_keepout',None)
                try:
                    place_pair(env,robot,estimator,updated,current,slots,report,alternatives,resample)
                finally:
                    robot.stationary_payload_keepout=old_keepout
        except RuntimeError as error:
            fail(str(error))
        finally:
            if action_start is not None:
                report['action_wall_s']=time.monotonic()-action_start
                report['action_planning_counts']=count_delta(action_planning_start)
                if timing is not None:report['action_timing']=timing.delta(action_timing_start)
                report['action_tick_counts']={k:v-action_ticks_start.get(k,0)
                    for k,v in getattr(robot,'tick_counts',{}).items()}
                report['physics_dt_s']=robot.dt
                report['action_validation_reuse']={k:v-validation_start.get(k,0) for k,v in getattr(robot,'validation_reuse',{}).items()}
                if reader is not None:
                    report['action_encoder_snapshots']=dict(acquisitions=reader.reads-encoder_start[0],
                        reuses=reader.reuses-encoder_start[1])
                report['observation_deduplication']=dict(getattr(estimator,'redundant_views',{}))
                for arm in states:
                    robot.command_velocities(arm,np.zeros(9))
                    robot.targets[arm][:7]=robot.read_states()[arm].joint_positions_rad
                env._write_json('graspgen_pair_check.json',report)
    report['success']=True
    env._write_json('graspgen_pair_check.json',report)
    return report
