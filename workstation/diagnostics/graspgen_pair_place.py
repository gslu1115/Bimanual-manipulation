"""Two independently grasped cartons, joint commands on a shared physics clock."""
import math
from dataclasses import replace
import numpy as np

from workstation.observations.camera_geometry import pose_matrix, quaternion_from_matrix
from workstation.perception.held_surface import CuboidHoldingTracker
from workstation.perception.preclose_target import PreCloseTarget, preclose_supported, reconcile_preclose_target
from workstation.perception.scene_estimate import ScenePriors
from workstation.planning.collision import AABB, sphere_box_clearance, check_spheres
from workstation.skills.cumotion_pick_place import placement_goal
from workstation.diagnostics.graspgen_pair import drive_empty_pregrasps, synchronized_knots


def transfer_schedule(robot,paths,payload_guard):
    """Choose a clear timing of the same two verified spatial paths before moving."""
    arms=('panda_left','panda_right')
    for delay,waiting in [(0.,0),(.35,0),(.35,1),(.7,0),(.7,1),(1.,0),(1.,1)]:
        scheduled=list(paths)
        if delay:
            for i,path in enumerate(paths):
                wait=[path[0] if i==waiting else path[-1]]*math.ceil((len(path)-1)*delay)
                scheduled[i]=wait+list(path) if i==waiting else list(path)+wait
        try:
            for left,right in synchronized_knots(*scheduled,max_step=.015):
                reason,_=check_spheres(robot.spheres(arms[0],left),robot.spheres(arms[1],right),[],None,
                                       margin=.008)
                if reason:raise RuntimeError(reason)
                payload_guard(left,right)
        except RuntimeError:
            continue
        return scheduled,dict(delayed_arm=arms[waiting] if delay else None,delay_fraction=delay)
    raise RuntimeError('NO_CLEAR_PAIRED_TRANSFER_SCHEDULE')


def place_pair(env, robot, estimator, scene, pair, slots, report):
    arms = ('panda_left', 'panda_right')
    priors = ScenePriors.from_config(env.c)
    selected = dict(zip(arms, pair['selection']))
    goals = dict(zip(arms, pair['slots']))
    objects = {a: next(o for o in scene.objects if o.track_id == selected[a]['target_id']) for a in arms}
    trackers = {}
    measurements = {}
    held = False
    phases = report['phases'] = {}

    def observe():
        nonlocal scene
        packet = env.get_observation_packet(refresh=True)
        scene = estimator.estimate(packet)
        robot.set_observation(packet, scene, priors)
        slots.observe(robot.observed_workspace)
        return packet

    def confirm_target(arm,packet,closed=False):
        from workstation.diagnostics.vision_check import save_observation
        proof=references[arm].confirm(packet,scene,closed=closed)
        report.setdefault('preclose_evidence',[]).append(proof)
        if not preclose_supported(proof,packet.observation_time_s):
            save_observation(env.output,'pair_'+arm+'_'+report['last_phase'],packet,scene)
            raise RuntimeError('PAIR_TARGET_UNSUPPORTED:'+arm+':'+proof['reason'])
        return proof

    def carried_clearance(left, right):
        boxes = []
        for arm, q, other, other_q in zip(arms, (left, right), reversed(arms), (right, left)):
            pos, rot = robot.fk(arm, q)
            t = trackers[arm]
            centre = pos + rot @ t.offset
            extent = np.abs(rot @ t.relative_rotation) @ (t.size + 2*t.uncertainty_m)
            box = AABB(objects[arm].track_id, centre, extent)
            if any(sphere_box_clearance(c, r, box) < .002 for _, c, r in robot.spheres(other, other_q)):
                raise RuntimeError('PAYLOAD_OTHER_ARM_COLLISION')
            boxes.append(box)
        a,b = boxes
        if np.all(np.abs(a.centre-b.centre) <= (a.size+b.size)/2+.002):
            raise RuntimeError('PAYLOAD_PAYLOAD_COLLISION')

    def measure_held(arm,packet):
        nonlocal scene
        t=trackers[arm]
        pos,rot=robot.fk(arm)
        observed=t.observe(packet,scene,pos,rot,reacquire_tilt=held)
        if observed is None:
            from workstation.diagnostics.vision_check import save_observation
            save_observation(env.output,'pair_held_'+arm+'_'+report['last_phase'],packet,scene)
            report['held_surface_failure']=dict(arm=arm,measurement=t.last_failure)
            raise RuntimeError('PAIR_HELD_SURFACE_UNCONFIRMED:'+arm)
        measurements[arm]=observed
        # The wrist can split the carried carton into new visual IDs. Reconcile
        # only the masks supported by this arm's current measured surfaces.
        scene=estimator.associate_supported_target(scene,objects[arm].track_id,observed)
        # Only a fully measured center can move the collision proxy away
        # from its previous reference. Keep the uncertainty of a partial fit.
        if observed.get('geometric_pose_measured'):
            t.offset=rot.T@(np.asarray(observed['position_m'])-pos)
            t.relative_rotation=rot.T@np.asarray(observed['rotation'])
        t.uncertainty_m=max(t.uncertainty_m,observed['uncertainty_m'])
        return observed

    def verify_slots(packet):
        # Reconcile BOTH carried cartons before testing the slot underneath
        # either one. Register the current arm-specific payload to explain its
        # shadow using saved free depth rays, never using ownership as clearance.
        for arm in arms:
            t=trackers[arm]
            robot.set_observation(packet,scene,priors)
            robot.held_visual_geometry=dict(arm=arm,target_id=objects[arm].track_id,offset=t.offset,
                relative_rotation=t.relative_rotation,uncertainty_m=t.uncertainty_m,allow_grasp_translation=True)
            if not robot.confirm_payload_observation(objects[arm].track_id,t,measurements[arm],packet):
                raise RuntimeError('PAIR_PAYLOAD_GEOMETRY_UNCONFIRMED:'+arm)
            xy=goals[arm]
            view=robot.observed_workspace.slot(xy)
            report.setdefault('destination_checks',[]).append(dict(arm=arm,phase=report['last_phase'],**view))
            if slots.owners.get(xy)!=arm or not view['free']:
                from workstation.diagnostics.vision_check import save_observation
                save_observation(env.output,'pair_slot_'+arm+'_'+report['last_phase'],packet,scene)
                raise RuntimeError('PAIR_DESTINATION_UNCONFIRMED:'+arm+':'+str(view['reason']))

    def move(phase, destinations, contact=False):
        report['last_phase'] = phase
        paths = []
        for arm in arms:
            o = objects[arm]
            q = robot.read_states()[arm].joint_positions_rad
            pos, rot = robot.fk(arm, q)
            goal, goal_rot = destinations[arm]
            path = robot.cartesian_knots(arm, pos, goal, 0., 0., .12, q,
                orientation0=quaternion_from_matrix(rot), orientation1=quaternion_from_matrix(goal_rot))
            if held:
                t = trackers[arm]
                robot.held_visual_geometry = dict(arm=arm, target_id=o.track_id, offset=t.offset,
                    relative_rotation=t.relative_rotation, uncertainty_m=t.uncertainty_m,allow_grasp_translation=True)
                if not robot.confirm_payload_observation(o.track_id,t,measurements[arm],packet):
                    raise RuntimeError('PAIR_PAYLOAD_GEOMETRY_UNCONFIRMED:'+arm)
            try:
                robot.validate_path(arm,path,scene,o.track_id, held_size=o.size_m if held else None,
                    support_contact=contact and held, target_contact=contact and not held,
                    check_local_unknown=not held and phase=='approach',check_approach_corridor=phase=='approach')
            except RuntimeError:
                from workstation.diagnostics.vision_check import save_observation
                save_observation(env.output,'pair_blocked_'+phase,packet,scene)
                report['path_failure']=dict(arm=arm, details=robot.last_plan_check,
                    robot_pixels_removed=estimator.last_robot_filter_counts.copy())
                raise
            paths.append(path)
        # Final Cartesian contact is verified below in metres/radians. The
        # per-step 0.04 rad stop and actual two-arm collision checks stay active.
        transfer=phase in ('transfer','align')
        if transfer:
            paths,schedule=transfer_schedule(robot,paths,carried_clearance)
            report[phase+'_schedule']=schedule
        phases[phase] = drive_empty_pregrasps(robot, paths, carried_clearance if held else None,
                                              following_band=.035 if contact else .025,require_overlap=not transfer,
                                              speed_fraction=.5 if transfer else 1.)
        for arm,(goal,goal_rot) in destinations.items():
            pos,rot=robot.fk(arm)
            error=float(np.linalg.norm(pos-goal))
            angle=float(np.arccos(np.clip((np.trace(rot.T@goal_rot)-1.)/2.,-1.,1.)))
            phases[phase].setdefault('tcp_errors',{})[arm]=dict(position_m=error,angle_rad=angle)
            if error>.005 or angle>math.radians(3):
                raise RuntimeError('PAIR_TCP_GOAL_MISSED:'+arm)

    def fingers(close):
        steps = math.ceil(.6/robot.dt)
        starts = {a:robot.read_states()[a].finger_positions_m.copy() for a in arms}
        for i in range(1,steps+1):
            for arm in arms:
                width = max(0., selected[arm]['closing_width_m']/2-.004) if close else .04
                robot.targets[arm][7:] = starts[arm]+(width-starts[arm])*i/steps
            robot.tick()
        robot.hold(.15)

    poses = {a:(np.asarray(c['position_m']),pose_matrix(c['position_m'],c['orientation_wxyz'])[:3,:3])
             for a,c in selected.items()}
    # The empty approach can outlive the visual reference horizon. Confirm
    # both original poses at the pregrasp waypoint before descending.
    references={a:PreCloseTarget(objects[a],scene,priors) for a in arms}
    report['last_phase']='pregrasp_recheck'
    packet=observe()
    for a in arms:
        prior=objects[a]
        matches=[o for o in scene.objects if o.posture=='UPRIGHT'
                 and not o.failures and np.linalg.norm(np.asarray(o.position_m)-prior.position_m)<=.002
                 and abs((o.axis_yaw_rad-prior.axis_yaw_rad+math.pi/2)%math.pi-math.pi/2)<=math.radians(3)
                 and np.allclose(o.size_m,prior.size_m,atol=1e-6,rtol=0)]
        if len(matches)==1:
            current=matches[0]
            if current.track_id!=prior.track_id:
                if any(o.track_id==prior.track_id for o in scene.objects):
                    raise RuntimeError('PAIR_AMBIGUOUS_PREGRASP_ID:'+a)
                report.setdefault('pregrasp_identity',{})[a]=dict(original=prior.track_id,current=current.track_id)
            references[a]=PreCloseTarget(current,scene,priors)
            objects[a]=current
        else:
            proof=confirm_target(a,packet)
            scene=reconcile_preclose_target(estimator,scene,proof)
            # A complete current 6-D fit (including all three translations)
            # can renew the stationary reference. A hidden-axis prior cannot.
            if proof['surface']['geometric_pose_measured']:
                measured=next(o for o in scene.objects if o.track_id==prior.track_id)
                same_face=replace(measured,posture=prior.posture,
                    hypotheses=tuple(h for h in measured.hypotheses if h.label==prior.posture))
                references[a]=PreCloseTarget(same_face,scene,priors)
    robot.set_observation(packet,scene,priors)
    move('approach', poses, True)
    packet = observe()
    for a in arms:
        confirm_target(a,packet)
    report['last_phase']='close'
    fingers(True)
    packet = observe()
    for a in arms:
        proof = confirm_target(a,packet,closed=True)
        scene=reconcile_preclose_target(estimator,scene,proof)
        trackers[a] = CuboidHoldingTracker.from_preclose_support(references[a].obj,proof,*robot.fk(a),priors)
        measure_held(a,packet)
    robot.set_observation(packet,scene,priors)
    held = True
    raised={a:(robot.fk(a)[0]+[0.,0.,.18],robot.fk(a)[1]) for a in arms}
    # Refit after initial finger loading; using the closure frame until the
    # end of an 18 cm lift loses track of cartons that rotate under load.
    move('initial_lift',{a:(p-[0.,0.,.12],r) for a,(p,r) in raised.items()},True)
    packet=observe()
    for a in arms:
        phases.setdefault('initial_lift_evidence',{})[a]=measure_held(a,packet)
    robot.set_observation(packet,scene,priors)
    move('lift',raised,True)
    packet = observe()
    places = {}
    for a in arms:
        t = trackers[a]
        support = measure_held(a,packet)
        if support is None or support['height_gain_m'] < .08:
            raise RuntimeError('PAIR_LIFT_UNCONFIRMED:'+a)
        phases.setdefault('lift_evidence',{})[a]=support
        places[a] = placement_goal(goals[a],robot.place_support_z,t.size,.012,
                                   robot.fk(a)[1]@t.relative_rotation,t.offset,t.relative_rotation)
    robot.set_observation(packet,scene,priors)
    report['simultaneous_grasp_verified']=phases['lift']['overlap_ticks']>0
    robot.anchor_observation()
    # Carry without simultaneously rotating the carton. Aim the BOX centre
    # above its lane using the measured grasp offset before changing attitude.
    overhead={}
    for a in arms:
        t=trackers[a]
        rot=robot.fk(a)[1]
        centre=np.array([*goals[a],robot.place_support_z+t.size[2]/2+.012+.18])
        overhead[a]=(centre-rot@t.offset,rot)
    move('transfer',overhead)
    packet=observe()
    for a in arms:
        observed=measure_held(a,packet)
        t=trackers[a]
        places[a]=placement_goal(goals[a],robot.place_support_z,t.size,.012,
                                 observed['rotation'],t.offset,t.relative_rotation)
    verify_slots(packet)
    robot.set_observation(packet,scene,priors)
    move('align',{a:(p+[0.,0.,.18],r) for a,(p,r) in places.items()})
    packet=observe()
    for a in arms:
        observed=measure_held(a,packet)
        t=trackers[a]
        places[a]=placement_goal(goals[a],robot.place_support_z,t.size,.012,
                                 observed['rotation'],t.offset,t.relative_rotation)
    verify_slots(packet)
    robot.set_observation(packet,scene,priors)
    move('lower',places,True)
    packet=observe()
    for a in arms:
        measurement=measure_held(a,packet)
        rotation=np.asarray(measurement['rotation'])
        height=np.asarray(measurement['position_m'])[2]-np.abs(rotation[2])@(trackers[a].size/2)
        upper_gap=height-robot.place_support_z+measurement['uncertainty_m']
        if upper_gap>.020:raise RuntimeError('PAIR_RELEASE_HEIGHT_UNSUPPORTED:'+a)
    report['last_phase']='release'
    fingers(False)
    held=False
    robot.held_visual_geometry=None
    observe()
    move('retreat',{a:(p+[0.,0.,.18],r) for a,(p,r) in places.items()},True)
    report['last_phase']='verify'
    stable_since=None
    previous_ids=None
    for _ in range(20):
        robot.hold(.1)
        observe()
        placements={}
        for a in arms:
            xy=goals[a]
            matches=[o for o in scene.objects if o.posture=='UPRIGHT' and not o.failures and
                     np.linalg.norm(np.array(o.position_m[:2])-xy)<=.018 and
                     abs(o.position_m[2]-robot.place_support_z-o.size_m[2]/2)<=.008 and
                     abs((o.axis_yaw_rad+math.pi/2)%math.pi-math.pi/2)<=math.radians(8) and
                     o.velocity_m_s is not None and np.linalg.norm(o.velocity_m_s)<=.015 and
                     np.allclose(o.size_m,objects[a].size_m,atol=.002,rtol=0)]
            if len(matches)==1:
                o,=matches
                placements[a]=dict(initial_track=selected[a]['target_id'],grasp_track=objects[a].track_id,observed_track=o.track_id,
                    goal_xy=xy,visual_error_xy_m=(np.asarray(o.position_m[:2])-xy).tolist())
        ids=tuple(p['observed_track'] for p in placements.values())
        if len(set(ids))!=2 or ids!=previous_ids:
            stable_since=scene.observation_time_s
        previous_ids=ids
        if len(set(ids))==2 and scene.observation_time_s-stable_since>=.5:
            for a,p in placements.items():
                slots.confirm_placement(a,goals[a],p['observed_track'])
            report['placements']=placements
            break
    else:raise RuntimeError('PAIR_RELEASED_CARTONS_NOT_STABLE')
    report['slots_final']=slots.report()
