"""Two independently grasped cartons, joint commands on a shared physics clock."""
import math
import numpy as np

from workstation.observations.camera_geometry import pose_matrix, quaternion_from_matrix
from workstation.perception.held_surface import HeldSurfaceTracker
from workstation.perception.preclose_target import PreCloseTarget, preclose_supported
from workstation.perception.scene_estimate import ScenePriors
from workstation.planning.collision import AABB, sphere_box_clearance
from workstation.skills.cumotion_pick_place import placement_goal
from workstation.diagnostics.graspgen_pair import drive_empty_pregrasps


def place_pair(env, robot, estimator, scene, pair, slots, report):
    arms = ('panda_left', 'panda_right')
    priors = ScenePriors.from_config(env.c)
    selected = dict(zip(arms, pair['selection']))
    goals = dict(zip(arms, pair['slots']))
    objects = {a: next(o for o in scene.objects if o.track_id == selected[a]['target_id']) for a in arms}
    references = {a: PreCloseTarget(o, scene, priors) for a, o in objects.items()}
    trackers = {}
    held = False
    phases = report['phases'] = {}

    def observe():
        nonlocal scene
        packet = env.get_observation_packet(refresh=True)
        scene = estimator.estimate(packet)
        robot.set_observation(packet, scene, priors)
        slots.observe(robot.observed_workspace)
        return packet

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

    def move(phase, destinations, contact=False):
        report['last_phase'] = phase
        paths = []
        for arm in arms:
            c, o = selected[arm], objects[arm]
            q = robot.read_states()[arm].joint_positions_rad
            pos, rot = robot.fk(arm, q)
            goal, goal_rot = destinations[arm]
            path = robot.cartesian_knots(arm, pos, goal, 0., 0., .12, q,
                orientation0=quaternion_from_matrix(rot), orientation1=quaternion_from_matrix(goal_rot))
            if held:
                t = trackers[arm]
                robot.held_visual_geometry = dict(arm=arm, target_id=o.track_id, offset=t.offset,
                    relative_rotation=t.relative_rotation, uncertainty_m=t.uncertainty_m)
            robot.validate_path(arm,path,scene,o.track_id, held_size=o.size_m if held else None,
                support_contact=contact and held, target_contact=contact and not held,
                check_local_unknown=not held and phase=='approach',check_approach_corridor=phase=='approach')
            paths.append(path)
        phases[phase] = drive_empty_pregrasps(robot, paths, carried_clearance if held else None)

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
    move('approach', poses, True)
    packet = observe()
    for a in arms:
        if not preclose_supported(references[a].confirm(packet,scene), packet.observation_time_s):
            raise RuntimeError('PAIR_PRECLOSE_UNSUPPORTED:'+a)
    report['last_phase']='close'
    fingers(True)
    packet = observe()
    for a in arms:
        proof = references[a].confirm(packet,scene,closed=True)
        if not preclose_supported(proof, packet.observation_time_s):
            raise RuntimeError('PAIR_POSTCLOSE_UNSUPPORTED:'+a)
        trackers[a] = HeldSurfaceTracker.from_preclose_support(proof,*robot.fk(a),packet.observation_time_s)
    held = True
    move('lift',{a:(robot.fk(a)[0]+[0.,0.,.18],robot.fk(a)[1]) for a in arms},True)
    packet = observe()
    places = {}
    for a in arms:
        t = trackers[a]
        support = t.observe(packet,scene,*robot.fk(a))
        if support is None or support['height_gain_m'] < .08:
            raise RuntimeError('PAIR_LIFT_UNCONFIRMED:'+a)
        phases.setdefault('lift_evidence',{})[a]=support
        places[a] = placement_goal(goals[a],robot.place_support_z,t.size,.012,
                                   robot.fk(a)[1]@t.relative_rotation,t.offset,t.relative_rotation)
    report['simultaneous_grasp_verified']=phases['lift']['overlap_ticks']>0
    move('transfer',{a:(p+[0.,0.,.18],r) for a,(p,r) in places.items()})
    packet=observe()
    for a in arms:
        if trackers[a].observe(packet,scene,*robot.fk(a)) is None:
            raise RuntimeError('PAIR_TRANSFER_UNCONFIRMED:'+a)
        if slots.states[goals[a]]!='reserved':
            raise RuntimeError('PAIR_DESTINATION_UNCONFIRMED:'+a)
    move('lower',places,True)
    packet=observe()
    for a in arms:
        if trackers[a].observe(packet,scene,*robot.fk(a)) is None:
            raise RuntimeError('PAIR_PRERELEASE_UNCONFIRMED:'+a)
    report['last_phase']='release'
    fingers(False)
    held=False
    robot.held_visual_geometry=None
    observe()
    move('retreat',{a:(p+[0.,0.,.18],r) for a,(p,r) in places.items()},True)
    robot.hold(.5)
    observe()
    report['placements']={}
    for a in arms:
        xy=goals[a]
        matches=[o for o in scene.objects if o.posture=='UPRIGHT' and not o.failures and
                 np.linalg.norm(np.array(o.position_m[:2])-xy)<=.018 and
                 np.allclose(o.size_m,objects[a].size_m,atol=.002,rtol=0)]
        if len(matches)!=1:raise RuntimeError('PAIR_RELEASED_CARTON_UNCONFIRMED:'+a)
        o,=matches
        slots.confirm_placement(a,xy,o.track_id)
        report['placements'][a]=dict(initial_track=objects[a].track_id,observed_track=o.track_id,
            goal_xy=xy,visual_error_xy_m=(np.asarray(o.position_m[:2])-xy).tolist())
    report['slots_final']=slots.report()
