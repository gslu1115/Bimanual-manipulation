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


def refine_pinch_contact(candidate,obj):
    """Fit a nearby model pinch to the observed parallel sides and pad depth."""
    from workstation.observations.camera_geometry import pose_matrix,quaternion_from_matrix
    from workstation.planning.graspgen_candidates import opening_from_visual_box
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


def preferred_assignments(targets, bases):
    """Assign both cartons to distinct arms with the shortest total XY reach."""
    left,right=sorted(bases)
    def distance(arm,obj):
        return np.linalg.norm(np.asarray(bases[arm]['base_xyz'])[:2]-np.asarray(obj.position_m)[:2])
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


def paired_pregrasp(robot, rows, states, scene):
    arms = ('panda_left', 'panda_right')
    feasible = [[a for a in rows if a.feasible and a.candidate.source == 'graspgen'
                 and a.candidate.arm == arm and a.terms.get('full_pose_checked')] for arm in arms]
    pairs = [(left, right) for left, right in product(*feasible)
             if left.candidate.target_id != right.candidate.target_id
             and ('slot_xy' not in left.terms or left.terms['slot_xy'] != right.terms['slot_xy'])]
    def slot_cost(pair):
        # Prefer independent lanes when free. The centre remains available
        # to either arm when a side lane is occupied or unreachable.
        return sum(abs(row.candidate.metadata.get('source_visual_object',{}).get(
                       'position_m',row.candidate.position_m)[0]-row.terms['slot_xy'][0])
                   for row in pair if row.terms.get('slot_xy') is not None)
    pairs.sort(key=lambda pair: (slot_cost(pair),-(pair[0].score+pair[1].score)))
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


def drive_empty_pregrasps(robot, paths, payload_guard=None, following_band=.025,speed_fraction=1.,
                          require_overlap=True,loaded_transfer=False,progress_check=None):
    """Commit both empty-arm targets before each shared physics step."""
    arms=('panda_left','panda_right')
    previous=[robot.read_states()[a].joint_positions_rad.copy() for a in arms]
    if any(np.max(np.abs(q-np.asarray(path[0])))>.01 for q,path in zip(previous,paths)):
        raise RuntimeError('PAIR_START_STATE_CHANGED')
    start=np.array(previous)
    commanded=start.copy()
    overlap=0
    ticks=0
    trace=[]
    def advance(desired):
        nonlocal previous,overlap,ticks,commanded
        # Also check tracking lag, not just the nominal synchronized paths.
        for left,right in synchronized_knots([previous[0],desired[0]],[previous[1],desired[1]]):
            if payload_guard:payload_guard(left,right)
            reason,_=check_spheres(robot.spheres(arms[0],left),robot.spheres(arms[1],right),[],None)
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
        robot.tick()
        states=robot.read_states()
        measured=np.array([states[a].joint_positions_rad for a in arms])
        if not np.isfinite(measured).all():
            raise RuntimeError('PAIR_JOINT_TRACKING_ERROR: nonfinite encoders')
        errors=np.abs(measured-desired)
        if np.max(errors)>.04:
            i,j=np.unravel_index(np.argmax(errors),errors.shape)
            raise RuntimeError(f'PAIR_JOINT_TRACKING_ERROR: {arms[i]} joint {j+1} '
                               f'error {errors[i,j]:.4f} rad at tick {ticks}')
        reason,_=check_spheres(robot.spheres(arms[0],measured[0]),robot.spheres(arms[1],measured[1]),[],None)
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
            advance(desired)
        else:
            errors=np.max(np.abs(previous-desired),axis=1)
            raise RuntimeError('PAIR_JOINT_TRACKING_ERROR: '+str(dict(zip(arms,errors.tolist()))))
    for _ in range(math.ceil(.3/robot.dt)):
        advance(np.asarray([p[-1] for p in paths]))
    moved=np.max(np.abs(previous-start),axis=1)
    if require_overlap and (not overlap or np.min(moved)<.04):raise RuntimeError('NO_MEASURED_PARALLEL_APPROACH')
    return dict(ticks=ticks,overlap_ticks=overlap,overlap_sim_s=overlap*robot.dt,
                arm_order=list(arms),joint_displacements_rad=moved.tolist(),trace=trace)


def check_pair(env, args, packet, scene, estimator):
    from workstation.diagnostics.graspgen_calibration import verify
    from workstation.models.graspgen_client import visual_clouds, infer_rotated_view
    from workstation.planning.selection import visual_pinch_balance
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
    armature_floor=.01 if execute else None
    robot=from_environment(env,paired_armature_floor=armature_floor)
    report['shoulder_drive_armature_floor_kg_m2']=armature_floor
    estimator.robot_point_filter=robot.observed_robot_points
    estimator.robot_filter_factory=robot.prepare_observed_robot_filter
    robot.set_observation(packet,scene,ScenePriors.from_config(env.c))
    slots = ConveyorSlots(env.c)
    destinations=[slots.positions[i] for i in env.c.get('pair_slot_indices',range(len(slots.positions)))]
    report['eligible_destinations']=destinations
    slots.observe(robot.observed_workspace)
    report['slots_initial'] = slots.report()
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
    def check_destination(c,o,s,slot):
        # Pick IK, empty approach and finger closure do not depend on the
        # downstream slot. Reuse their failure only for this frozen snapshot.
        if c.candidate_id in pick_failures:return pick_failures[c.candidate_id]
        result=robot.feasibility(c,o,s,slot)
        if result.get('reasons') and result.get('rejection_phase') in ('PREGRASP','APPROACH','CLOSE'):
            pick_failures[c.candidate_id]=result
        return result
    for attempt in range(1,8):
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
                targets=[o for o in scene.objects if o.posture=='UPRIGHT' and not o.failures and pose_is_observed(o,scene)]
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
                return offset+abs(depth-.45)+abs(opening['centre_offset_m'])/.02
            for arm in states:
                if attempt<=4 and arm!=preferred.get(obj.track_id):continue
                ordered=sorted((c for c in allowed if c.arm==arm),key=pinch_cost)
                candidates.setdefault(obj.track_id,[]).extend(ordered[:4])
        for slot in available:
            scored=assess(scene,states,lambda c,o,s:check_destination(c,o,s,slot),robot.home,
                          candidate_provider=lambda o,_:candidates.get(o.track_id,[]))
            for row in scored:
                row.terms['slot_xy']=slot
            rows.extend(scored)
            for arm in states:
                assigned=[r for r in scored if r.candidate.arm==arm]
                report['slot_assessments'].append(dict(attempt=attempt,slot_xy=slot,arm=arm,
                    feasible=sum(r.feasible for r in assigned),
                    feasible_by_target=dict(Counter(r.candidate.target_id for r in assigned if r.feasible)),
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
                renewed=renew_view_candidate(c,updated,seq)
                if renewed.target_id!=c.target_id:
                    report.setdefault('reidentified_after_view',{})[c.target_id]=renewed.target_id
                new[renewed.target_id]=[renewed]
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
                place_pair(env,robot,estimator,updated,current,slots,report)
        except RuntimeError as error:
            fail(str(error))
        finally:
            if action_start is not None:
                report['action_wall_s']=time.monotonic()-action_start
                for arm in states:
                    robot.command_velocities(arm,np.zeros(9))
                    robot.targets[arm][:7]=robot.read_states()[arm].joint_positions_rad
                env._write_json('graspgen_pair_check.json',report)
    report['success']=True
    env._write_json('graspgen_pair_check.json',report)
