"""Two independently grasped cartons, joint commands on a shared physics clock."""
import math
from dataclasses import replace, asdict
import numpy as np

from workstation.observations.camera_geometry import pose_matrix, quaternion_from_matrix
from workstation.perception.held_surface import CuboidHoldingTracker, HeldSurfaceTracker
from workstation.perception.preclose_target import PreCloseTarget, preclose_supported, reconcile_preclose_target
from workstation.perception.scene_estimate import ScenePriors, pose_is_observed
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


def stationary_slot_tracks(workspace, scene, positions):
    """Identify a single observed, supported, settled carton per occupied slot."""
    tracks={}
    for xy in positions:
        ids=workspace.slot(xy)['observed_objects']
        if len(ids)!=1:continue
        obj=next((o for o in scene.objects if o.track_id==ids[0]),None)
        if (obj is not None and not obj.failures and obj.posture=='UPRIGHT' and pose_is_observed(obj,scene)
                and obj.velocity_m_s is not None and np.linalg.norm(obj.velocity_m_s)<=.015
                and abs(obj.position_m[2]-workspace.priors.support_z_m-obj.size_m[2]/2)<=.008):
            tracks[xy]=obj
    return tracks


def parked_surface_envelope(packet,scene,reference,fragments,priors,goals):
    """Enclose all centres consistent with a clipped stationary carton at its observed lane."""
    from workstation.perception.cuboid_pose import fit_cuboid_pose, measured_cuboid_supported, reference_constrained_cuboid_supported
    from workstation.perception.rgbd_cartons import deproject
    yaw=reference['axis_yaw_rad'];c,s=math.cos(yaw),math.sin(yaw)
    prior_rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    size=np.asarray(reference['size_m'])
    fit=fit_cuboid_pose(packet,scene,reference['position_m'],prior_rotation,size,priors,
                        allow_reference_translation=True)
    masks=[ref for obj in fragments for ref in obj.mask_refs]
    if (not masks or len(set(masks))!=len(masks) or len({r.split('/')[0] for r in masks})!=len(masks)):return None
    full=measured_cuboid_supported(fit,packet.observation_time_s)
    if full or reference_constrained_cuboid_supported(fit,packet.observation_time_s):
        if set(fit['mask_refs'])!=set(masks):return None
        rotation=np.asarray(fit['rotation']);centre=np.asarray(fit['position_m'])
        if full:
            if np.linalg.norm(centre-reference['position_m'])>.005:return None
            axis=None
        else:
            if len(fit['observed_face_axes'])<2:return None
            axis,=fit['predicted_translation_axes']
        margin=fit['uncertainty_m']
        face_axes=fit['observed_face_axes']
    else:
        side=parked_side_with_top(packet,scene,reference,masks,priors)
        if side is None:return None
        centre,rotation,axis,margin=side
        face_axes=[1-axis,2]
    extent=size.copy();interval=None
    if axis is not None:
        values=[]
        for ref in masks:
            frame=packet.cameras[ref.split('/')[0]]
            points=deproject(frame)[scene.masks[ref]&frame.valid_depth]
            values.append((points-centre)@rotation[:,axis])
        low,high=np.quantile(np.concatenate(values),[.002,.998])
        lower,upper=high-size[axis]/2-margin,low+size[axis]/2+margin
        if lower>upper or lower>margin or upper < -margin:return None
        centre=centre+rotation[:,axis]*(lower+upper)/2
        extent[axis]+=upper-lower
        interval=[lower,upper]
    world=np.abs(rotation)@(extent+2*margin)
    # A proven X separation from BOTH destination corridors, including the
    # 90 mm slot query radius, even at the extremes of the hidden coordinate.
    if any(abs(centre[0]-g[0])<=world[0]/2+.09 for g in goals):return None
    obj=fragments[0]
    bounded=replace(obj,position_m=tuple(centre),axis_yaw_rad=0.,posture='UNKNOWN',mask_refs=tuple(masks),
        visibility='PARKED_COLLISION_ENVELOPE',size_m=tuple(world),hypotheses=(),measured_cuboid=None,
        quality={**obj.quality,'geometric_length_m':world[0],'geometric_width_m':world[1],
                 'observed_height_m':world[2]})
    return bounded,dict(track_id=obj.track_id,fragment_tracks=[o.track_id for o in fragments],
        time_s=packet.observation_time_s,mask_refs=masks,
        observed_face_axes=face_axes,bounded_axis=axis,offset_interval_m=interval,
        collision_center_m=centre.tolist(),collision_size_m=world.tolist())


def parked_side_with_top(packet,scene,reference,refs,priors):
    """A whole vertical side patch and a separate camera's top returns bound a parked obstacle."""
    from workstation.perception.cuboid_pose import _plane_patch
    from workstation.perception.rgbd_cartons import deproject
    if (scene.observation_time_s!=packet.observation_time_s or
            not 0<=packet.assembled_time_s-packet.observation_time_s<=priors.max_age_s):return None
    if len(refs)!=1:return None
    ref,=refs;name=ref.split('/')[0];frame=packet.cameras.get(name)
    if frame is None or packet.camera_status.get(name)!='OK' or frame.sample_time_s!=packet.observation_time_s:return None
    mask=scene.masks[ref]
    if not mask.any() or frame.valid_depth[mask].mean()<.95:return None
    points=deproject(frame)[mask&frame.valid_depth]
    centre=np.array(reference['position_m']);size=np.asarray(reference['size_m'])
    yaw=reference['axis_yaw_rad'];c,s=math.cos(yaw),math.sin(yaw)
    prior=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    local=(points-centre)@prior
    margin=max(.004,priors.optical_depth_error_margin_m)
    if np.mean(np.all(np.abs(local)<=size/2+margin,axis=1))<.95:return None
    for axis in (0,1):
        fit=_plane_patch(points,prior[:,axis])
        if fit is None:continue
        normal,residual=fit
        if residual>.001 or abs(normal[2])>.035 or normal@prior[:,axis]<math.cos(math.radians(3)):continue
        direction=normal.copy();direction[2]=0;direction/=np.linalg.norm(direction)
        up=np.array([0.,0.,1.])
        rotation=(np.column_stack((direction,np.cross(up,direction),up)) if axis==0 else
                  np.column_stack((np.cross(direction,up),direction,up)))
        offsets=(points-centre)@rotation
        level=np.median(offsets[:,axis])
        if abs(abs(level)-size[axis]/2)>margin:continue
        if np.any(np.ptp(np.quantile(offsets[:,[1-axis,2]],[.05,.95],axis=0),axis=0)<.012):continue
        if not parked_top_still_visible(packet,reference,exclude_camera=name):continue
        return centre+direction*(level-np.sign(level)*size[axis]/2),rotation,1-axis,margin
    return None


def parked_top_still_visible(packet,reference,mask_refs=(),masks=None,*,exclude_camera=None):
    """Require a connected depth patch at the previously observed top inside its footprint."""
    import cv2
    from workstation.perception.rgbd_cartons import deproject
    centre=np.asarray(reference['position_m']);size=np.asarray(reference['size_m'])
    yaw=reference['axis_yaw_rad'];c,s=math.cos(yaw),math.sin(yaw)
    rz=np.array([[c,-s],[s,c]])
    confirmed=False
    selected={ref.split('/')[0]:ref for ref in mask_refs}
    if len(selected)!=len(mask_refs):return False
    if any(name not in packet.cameras for name in selected):return False
    for name,f in packet.cameras.items():
        if name==exclude_camera:continue
        if selected and name not in selected:continue
        if f is None or packet.camera_status.get(name)!='OK' or f.sample_time_s!=packet.observation_time_s:
            if selected:return False
            continue
        xyz=deproject(f)
        local=(xyz[:,:,:2]-centre[:2])@rz
        mask=(f.valid_depth & np.all(np.abs(local)<=size[:2]/2,axis=2)
              & (np.abs(xyz[:,:,2]-centre[2]-size[2]/2)<.001))
        if selected:
            instance=masks[selected[name]]
            if not instance.any() or f.valid_depth[instance].mean()<.9:return False
            observed=np.column_stack((local[instance&f.valid_depth],xyz[:,:,2][instance&f.valid_depth]-centre[2]))
            residual=np.abs(observed)-size/2
            if np.mean((residual.max(axis=1)<=.004)&(np.abs(residual).min(axis=1)<=.004))<.95:return False
            mask&=instance
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
        for index in range(1,count):
            if stats[index,cv2.CC_STAT_AREA]<9:continue
            patch=xyz[labels==index]
            if np.ptp(np.quantile(patch[:,2],[.05,.95]))<.001 and np.min(np.ptp(patch[:,:2],axis=0))>.003:
                confirmed=True
    return confirmed


def parked_top_occluded(packet,reference):
    """All valid rays into the inner top footprint must terminate in front of it."""
    u,v=np.meshgrid(np.linspace(-.42,.42,11),np.linspace(-.42,.42,9))
    size=np.asarray(reference['size_m']);yaw=reference['axis_yaw_rad']
    c,s=math.cos(yaw),math.sin(yaw)
    local=np.column_stack((u.ravel()*size[0],v.ravel()*size[1]))
    world=np.tile(reference['position_m'],(u.size,1))
    world[:,:2]+=local@np.array([[c,s],[-s,c]])
    world[:,2]+=size[2]/2
    views=0
    for name,f in packet.cameras.items():
        if f is None or packet.camera_status.get(name)!='OK' or f.sample_time_s!=packet.observation_time_s:return False
        camera=(world-f.T_workcell_from_camera_cv[:3,3])@f.T_workcell_from_camera_cv[:3,:3]
        uv=camera@f.K.T
        pixels=np.rint(uv[:,:2]/np.maximum(camera[:,2:],1e-8)).astype(int)
        x,y=pixels.T;h,w=f.depth_m.shape
        inside=(camera[:,2]>0)&(x>=0)&(x<w)&(y>=0)&(y<h)
        if not inside.any():continue
        x,y=x[inside],y[inside]
        if not f.valid_depth[y,x].all() or np.any(f.depth_m[y,x]>=camera[inside,2]-.008):return False
        views+=int(inside.all())
    return views>=2


def place_pair(env, robot, estimator, scene, pair, slots, report,alternatives=None,resample=None):
    arms = ('panda_left', 'panda_right')
    priors = ScenePriors.from_config(env.c)
    selected = dict(zip(arms, pair['selection']))
    goals = dict(zip(arms, pair['slots']))
    unused=[xy for xy in slots.positions if xy not in goals.values()]
    parked={tuple(p['xy']):p for p in report.get('parked_before_pick',[])}
    initially_occupied={tuple(s['xy']) for s in report['slots_initial'] if s['state']=='occupied'}
    partial_keepout={}
    objects = {a: next(o for o in scene.objects if o.track_id == selected[a]['target_id']) for a in arms}
    trackers = {}
    release_trackers = {}
    measurements = {}
    held = False
    before_transfer={}
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
            for keepout,_ in partial_keepout.values():
                if (np.all(np.abs(centre-keepout.centre)<=(extent+keepout.size)/2+.008) or
                        any(sphere_box_clearance(p,r,keepout)<.008 for _,p,r in robot.spheres(arm,q))):
                    raise RuntimeError('PARKED_CARTON_CLEARANCE')
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
        if observed is None and report['last_phase'] in ('initial_lift','regrasp_initial_lift'):
            # An already slipping carton can leave the nominal FK search band.
            # Search down along gravity only during the test lift. Two current
            # views must independently supply masks on the fitted cuboid.
            from workstation.perception.held_surface import reacquire_tilted_top
            hint=pos+rot@t.offset
            for drop in (.010,.020):
                fit=reacquire_tilted_top(packet,scene,hint-[0,0,drop],rot@t.relative_rotation,
                    t.size,priors,allow_grasp_translation=True)
                if fit is None or len({ref.split('/')[0] for ref in fit['mask_refs']})<2:continue
                delta=np.asarray(fit['position_m'])-hint
                if not -.027<=delta[2]<0 or np.linalg.norm(delta[:2])>.008:continue
                fit['height_gain_m']=fit['position_m'][2]-t.initial_z
                fit['test_lift_search_drop_m']=drop
                observed=fit
                break
        if observed is None and report['last_phase'] in ('align','lower','regrasp_lower'):
            # A carton may settle onto the stationary belt before the hand
            # opens. Search at support height, keeping the grasp's XY prior:
            # the desired slot centre must never supply a hidden coordinate.
            from workstation.perception.cuboid_pose import fit_cuboid_pose,measured_cuboid_supported,grasp_constrained_cuboid_supported
            from workstation.planning.selection import nearest_upright_slot_rotation
            hint=pos+rot@t.offset
            upright=rot@t.relative_rotation if report['last_phase']=='regrasp_lower' else nearest_upright_slot_rotation(rot@t.relative_rotation)
            waypoint_height=.18 if report['last_phase']=='align' else 0.
            support_z=priors.support_z_m if report['last_phase']=='regrasp_lower' else robot.place_support_z
            for gap in (0.,.006,.012):
                centre=np.array([*hint[:2],support_z+t.size[2]/2+waypoint_height+gap])
                fit=fit_cuboid_pose(packet,scene,centre,upright,t.size,priors,allow_grasp_translation=True)
                if not (measured_cuboid_supported(fit,packet.observation_time_s) or
                        grasp_constrained_cuboid_supported(fit,packet.observation_time_s)):continue
                if np.linalg.norm(np.asarray(fit['position_m'])-hint)>.025:continue
                observed=fit
                observed['waypoint_reacquisition']=report['last_phase']
                break
        if observed is None:
            from workstation.diagnostics.vision_check import save_observation
            save_observation(env.output,'pair_held_'+arm+'_'+report['last_phase'],packet,scene)
            report['held_surface_failure']=dict(arm=arm,measurement=t.last_failure)
            raise RuntimeError('PAIR_HELD_SURFACE_UNCONFIRMED:'+arm)
        measurements[arm]=observed
        # The wrist can split the carried carton into new visual IDs. Reconcile
        # only the masks supported by this arm's current measured surfaces.
        scene=estimator.associate_supported_target(scene,objects[arm].track_id,observed)
        # Replace a clipped neighbour proxy only when its whole set of masks
        # is supported by this current carried-box fit. Keep other obstacles.
        target_id=objects[arm].track_id
        for obj in scene.objects:
            if obj.track_id!=target_id or set(obj.mask_refs)!=set(observed['mask_refs']):continue
            rotation=np.asarray(observed['rotation'])
            yaw=math.atan2(rotation[1,0],rotation[0,0])
            c,s=math.cos(yaw),math.sin(yaw)
            level=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            extent=np.abs(level.T@rotation)@t.size
            bounded=replace(obj,position_m=tuple(observed['position_m']),axis_yaw_rad=yaw,
                uncertainty_m=observed['uncertainty_m'],size_m=tuple(t.size),
                visibility='HELD_COLLISION_ENVELOPE',posture='UNKNOWN',failures=(),hypotheses=(),
                measured_cuboid=None,quality={**obj.quality,'geometric_length_m':extent[0],
                    'geometric_width_m':extent[1],'observed_height_m':extent[2]})
            scene=replace(scene,objects=tuple(bounded if o.track_id==target_id else o for o in scene.objects))
            break
        # Only a fully measured center can move the collision proxy away
        # from its previous reference. Keep the uncertainty of a partial fit.
        if observed.get('geometric_pose_measured'):
            t.offset=rot.T@(np.asarray(observed['position_m'])-pos)
        # The partial fit still measures all rotation axes. One hidden
        # translation must not freeze the measured carton tilt at closure.
        if observed.get('geometric_pose_measured') or observed.get('grasp_constrained'):
            t.relative_rotation=rot.T@np.asarray(observed['rotation'])
        t.uncertainty_m=max(t.uncertainty_m,observed['uncertainty_m'])
        return observed

    def observe_destinations():
        packet=observe()
        if initially_occupied and stationary_slot_tracks(robot.observed_workspace,scene,unused).keys()!=initially_occupied:
            first=packet.observation_time_s
            robot.hold(.1)
            packet=observe()
            report.setdefault('parked_rechecks',[]).append(dict(phase=report['last_phase'],
                first_s=first,second_s=packet.observation_time_s))
        return packet

    def parked_unchanged(current,enclosed=()):
        return ((current.keys()|set(enclosed))==parked.keys()==initially_occupied and
                all(np.linalg.norm(np.subtract(o.position_m,parked[k]['position_m']))<=.005 and
                    np.allclose(o.size_m,parked[k]['size_m'],atol=.002,rtol=0) for k,o in current.items()))

    def verify_slots(packet):
        nonlocal scene
        # Reconcile BOTH carried cartons before testing the slot underneath
        # either one. Register the current arm-specific payload to explain its
        # shadow using saved free depth rays, never using ownership as clearance.
        current=stationary_slot_tracks(robot.observed_workspace,scene,unused)
        parked_ids={o.track_id for o in current.values()}
        carried_ids={objects[a].track_id for a in arms}
        enclosed=set()
        if env.conveyor.state=='WAITING' and not env.safe_to_index:
            if report['last_phase']=='transfer' and parked_unchanged(current):
                for xy,o in current.items():
                    c,s=math.cos(o.axis_yaw_rad),math.sin(o.axis_yaw_rad)
                    R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                    box=AABB('parked_carton',np.asarray(o.position_m),np.abs(R)@np.asarray(o.size_m)+.016)
                    partial_keepout[xy]=(box,packet.observation_time_s)
                    robot.stationary_payload_keepout=box
            for xy in parked.keys()-current.keys():
                ids=robot.observed_workspace.slot(xy)['observed_objects']
                fragments=[o for o in scene.objects if o.track_id in ids]
                if carried_ids.intersection(ids):continue
                if xy in partial_keepout and report['last_phase']=='align':
                    box,seen=partial_keepout[xy]
                    refs=[ref for obj in fragments for ref in obj.mask_refs]
                    fresh=0<packet.observation_time_s-seen<=8.
                    patch=fresh and parked_top_still_visible(packet,parked[xy],refs,scene.masks)
                    occluded=fresh and not ids and not patch and parked_top_occluded(packet,parked[xy])
                    if patch or occluded:
                        enclosed.add(xy)
                        parked_ids.update(ids)
                        report.setdefault('parked_occlusion_checks',[]).append(dict(
                            xy=list(xy),mask_refs=refs,time_s=packet.observation_time_s,last_surface_s=seen,
                            source=('current top-depth patch' if patch else 'foreground occlusion in current depth views')
                                   +' and continuously protected collision envelope'))
                        continue
                if not fragments:continue
                bound=parked_surface_envelope(packet,scene,parked[xy],fragments,priors,goals.values())
                if bound is None:continue
                proxy,proof=bound
                box=AABB('parked_carton',np.asarray(proxy.position_m),np.asarray(proxy.size_m))
                partial_keepout[xy]=(box,packet.observation_time_s)
                robot.stationary_payload_keepout=box
                scene=replace(scene,objects=tuple(o for o in scene.objects if o.track_id not in ids)+(proxy,))
                report.setdefault('parked_surface_bounds',[]).append(proof)
                parked_ids.add(proxy.track_id);enclosed.add(xy)
        supported=parked_unchanged(current,enclosed)
        for arm in arms:
            t=trackers[arm]
            robot.set_observation(packet,scene,priors)
            robot.held_visual_geometry=dict(arm=arm,target_id=objects[arm].track_id,offset=t.offset,
                relative_rotation=t.relative_rotation,uncertainty_m=t.uncertainty_m,allow_grasp_translation=True)
            if not robot.confirm_payload_observation(objects[arm].track_id,t,measurements[arm],packet):
                raise RuntimeError('PAIR_PAYLOAD_GEOMETRY_UNCONFIRMED:'+arm)
            xy=goals[arm]
            view=robot.observed_workspace.slot(xy)
            accounted=(len(carried_ids)==2 and supported and carried_ids.isdisjoint(parked_ids) and
                       {o.track_id for o in scene.objects}==carried_ids|parked_ids and
                       all(measurements[a]['time_s']==packet.observation_time_s and
                           measurements[a]['position_m'][2]-np.linalg.norm(trackers[a].size)/2>
                           robot.place_support_z+.04 for a in arms))
            evidence=slots.approach_clearance(arm,xy,view,before_transfer.get(xy),accounted_for=accounted,
                stationary_belt=env.conveyor.state=='WAITING' and not env.safe_to_index)
            report.setdefault('destination_checks',[]).append(dict(arm=arm,phase=report['last_phase'],
                evidence=evidence,prior=before_transfer.get(xy),both_cartons_accounted_for=accounted,
                observed_parked_tracks=sorted(parked_ids),**view))
            if evidence is None:
                from workstation.diagnostics.vision_check import save_observation
                save_observation(env.output,'pair_slot_'+arm+'_'+report['last_phase'],packet,scene)
                raise RuntimeError('PAIR_DESTINATION_UNCONFIRMED:'+arm+':'+str(view['reason']))

    def refresh_approach_references(elapsed,active):
        nonlocal packet,scene,next_approach_check
        if elapsed < next_approach_check:
            return
        next_approach_check=elapsed+3.
        packet=observe()
        for arm in active:
            prior=references[arm]
            proof=prior.confirm(packet,scene)
            if preclose_supported(proof,packet.observation_time_s) and proof['full_pose_measured']:
                scene=reconcile_preclose_target(estimator,scene,proof)
                references[arm]=prior.renewed(scene,proof)
            report.setdefault('approach_reference_checks',[]).append(dict(
                arm=arm,time_s=packet.observation_time_s,reason=proof['reason'],
                renewed=references[arm] is not prior))
        robot.set_observation(packet,scene,priors)

    def move(phase, destinations, contact=False,*,active=arms,carrying=None):
        nonlocal next_approach_check
        report['last_phase'] = phase
        paths = []
        for arm in arms:
            o = objects[arm]
            q = robot.read_states()[arm].joint_positions_rad
            if arm not in active:
                paths.append([q,q])
                continue
            carries=held if carrying is None else arm in carrying
            pos, rot = robot.fk(arm, q)
            goal, goal_rot = destinations[arm]
            try:
                path = robot.cartesian_knots(arm, pos, goal, 0., 0., .12, q,
                    orientation0=quaternion_from_matrix(rot), orientation1=quaternion_from_matrix(goal_rot))
            except RuntimeError as exc:
                if phase!='approach' or str(exc)!='IK_DISCONTINUITY':raise
                # Neither arm has started the descent. Check a nearby alternate
                # from the actual pregrasp FK against the current target.
                from workstation.planning.graspgen_candidates import revalidate_candidate
                seq={n:f.sequence_id for n,f in packet.cameras.items() if f}
                for alt in (alternatives or {}).get(arm,()):
                    if alt.candidate_id==selected[arm]['candidate_id']:continue
                    try:
                        alt=revalidate_candidate(replace(alt,target_id=o.track_id),o,scene,seq)
                        orientation=pose_matrix(alt.position_m,alt.orientation_wxyz)[:3,:3]
                        # Stay on the model's approach line. Move to its own
                        # pregrasp first, then descend only along that line.
                        pre=np.asarray(alt.position_m)-np.asarray(alt.approach_direction)*alt.pregrasp_distance_m
                        prefix=robot.cartesian_knots(arm,pos,pre,0.,0.,.12,q,
                            orientation0=quaternion_from_matrix(rot),orientation1=alt.orientation_wxyz)
                        robot.validate_path(arm,prefix,scene,o.track_id)
                        descent=robot.cartesian_knots(arm,pre,alt.position_m,0.,0.,.12,prefix[-1],
                            orientation0=alt.orientation_wxyz,orientation1=alt.orientation_wxyz)
                        robot.validate_path(arm,descent,scene,o.track_id,target_contact=True)
                    except RuntimeError:continue
                    report.setdefault('preclose_substitutions',[]).append(dict(arm=arm,
                        previous=selected[arm]['candidate_id'],replacement=alt.candidate_id,reason=str(exc)))
                    selected[arm]=asdict(alt)
                    pair['selection'][arms.index(arm)]=selected[arm]
                    destinations[arm]=(np.asarray(alt.position_m),orientation)
                    path=prefix+descent[1:]
                    break
                else:raise
            if carries:
                t = trackers[arm]
                robot.held_visual_geometry = dict(arm=arm, target_id=o.track_id, offset=t.offset,
                    relative_rotation=t.relative_rotation, uncertainty_m=t.uncertainty_m,allow_grasp_translation=True)
                if not robot.confirm_payload_observation(o.track_id,t,measurements[arm],packet):
                    raise RuntimeError('PAIR_PAYLOAD_GEOMETRY_UNCONFIRMED:'+arm)
            try:
                robot.validate_path(arm,path,scene,o.track_id, held_size=o.size_m if carries else None,
                    support_contact=contact and carries, target_contact=contact and not carries,
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
        both_loaded=held and (carrying is None or set(carrying)==set(arms))
        renew_reference=phase in ('approach','regrasp_pregrasp','regrasp_approach')
        if renew_reference:next_approach_check=3.
        phases[phase] = drive_empty_pregrasps(robot, paths, carried_clearance if both_loaded else None,
                                              following_band=.035 if contact else .025,require_overlap=not transfer and len(active)==2,
                                              speed_fraction=.4 if transfer else 1.,loaded_transfer=transfer,
                                              progress_check=(lambda elapsed:refresh_approach_references(elapsed,active)) if renew_reference else None)
        for arm,(goal,goal_rot) in destinations.items():
            pos,rot=robot.fk(arm)
            error=float(np.linalg.norm(pos-goal))
            angle=float(np.arccos(np.clip((np.trace(rot.T@goal_rot)-1.)/2.,-1.,1.)))
            phases[phase].setdefault('tcp_errors',{})[arm]=dict(position_m=error,angle_rad=angle)
            if error>.005 or angle>math.radians(3):
                raise RuntimeError('PAIR_TCP_GOAL_MISSED:'+arm)

    def fingers(close,active=arms):
        steps = math.ceil(.6/robot.dt)
        starts = {a:robot.read_states()[a].finger_positions_m.copy() for a in arms}
        for i in range(1,steps+1):
            for arm in active:
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
    next_approach_check=3.
    move('approach', poses, True)
    packet = observe()
    for a in arms:
        proof=confirm_target(a,packet)
        if proof['full_pose_measured']:
            scene=reconcile_preclose_target(estimator,scene,proof)
            references[a]=references[a].renewed(scene,proof)
    report['last_phase']='close'
    fingers(True)
    packet = observe()
    closed_states=robot.read_states()
    closed_tcp_z={a:robot.fk(a)[0][2] for a in arms}
    for a in arms:
        proof = confirm_target(a,packet,closed=True)
        scene=reconcile_preclose_target(estimator,scene,proof)
        trackers[a] = CuboidHoldingTracker.from_preclose_support(references[a].obj,proof,*robot.fk(a),priors,packet=packet)
        release_trackers[a]=HeldSurfaceTracker.from_preclose_support(proof,*robot.fk(a),packet.observation_time_s)
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
    from workstation.planning.selection import lift_slip_evidence
    def slip(a):
        return lift_slip_evidence(closed_tcp_z[a],robot.fk(a)[0][2],measurements[a],
            robot.read_states()[a].finger_positions_m,closed_states[a].finger_positions_m)
    checks=report['initial_grip_checks']={a:[slip(a)] for a in arms}
    suspect=[a for a in arms if checks[a][0]['suspect']]
    if suspect:
        robot.hold(.12)
        packet=observe()
        for a in arms:
            measure_held(a,packet)
            checks[a].append(slip(a))
        confirmed=[a for a in suspect if checks[a][-1]['suspect']]
        if len(confirmed)>1:raise RuntimeError('BOTH_GRIPS_SLIPPING')
        if confirmed:
            a=confirmed[0]
            other=next(b for b in arms if b!=a)
            recovery=report['one_arm_regrasp']=dict(arm=a,parked_arm=other,success=False,
                original=selected[a]['candidate_id'])
            # The other gripper stays closed at the measured first-lift waypoint.
            # Its observed carton remains a separate obstacle for the moving arm.
            m=measurements[other]
            robot.stationary_payload_keepout=AABB(objects[other].track_id,np.asarray(m['position_m']),
                np.abs(m['rotation'])@(trackers[other].size+2*m['uncertainty_m']))
            try:
                m=measurements[a]
                bottom=m['position_m'][2]-np.abs(np.asarray(m['rotation'])[2])@(trackers[a].size/2)
                tcp,rot=robot.fk(a)
                dz=bottom-priors.support_z_m-.004
                if not 0<dz<.08:raise RuntimeError('REGRASP_SUPPORT_HEIGHT_UNCONFIRMED')
                robot.set_observation(packet,scene,priors)
                move('regrasp_lower',{a:(tcp-[0,0,dz],rot)},True,active=(a,))
                packet=observe()
                m=measure_held(a,packet)
                gap=m['position_m'][2]-np.abs(np.asarray(m['rotation'])[2])@(trackers[a].size/2)-priors.support_z_m
                if not -.006<=gap<=.020-m['uncertainty_m']:raise RuntimeError('REGRASP_RELEASE_HEIGHT_UNCONFIRMED')
                fingers(False,(a,))
                tcp,rot=robot.fk(a)
                packet=observe()
                released=release_trackers[a].collision_estimate(packet,scene,*robot.fk(a),
                    max_drop_m=max(0.,gap+m['uncertainty_m']),reference_position_m=m['position_m'],
                    reference_rotation=m['rotation'],allow_partial=True)
                if released is None:
                    from workstation.perception.cuboid_pose import fit_cuboid_pose,reference_constrained_cuboid_supported,measured_cuboid_supported
                    centre=np.asarray(m['position_m']).copy()
                    centre[2]=priors.support_z_m+trackers[a].size[2]/2
                    fit=fit_cuboid_pose(packet,scene,centre,np.asarray(m['rotation']),trackers[a].size,priors,
                        allow_reference_translation=True)
                    if (measured_cuboid_supported(fit,packet.observation_time_s) or
                            reference_constrained_cuboid_supported(fit,packet.observation_time_s)):
                        released=fit
                if released is None:raise RuntimeError('REGRASP_RELEASED_TARGET_UNCONFIRMED')
                scene=estimator.associate_supported_target(scene,objects[a].track_id,released)
                if not any(o.track_id==objects[a].track_id and set(o.mask_refs)==set(released['mask_refs'])
                           for o in scene.objects):raise RuntimeError('REGRASP_RELEASED_TRACK_UNRESOLVED')
                robot.set_observation(packet,scene,priors)
                move('regrasp_retreat',{a:(tcp+[0,0,.16],rot)},True,active=(a,),carrying=(other,))
                robot.hold(.12)
                packet=observe()
                # The retreat image may introduce a fresh visual ID. A second
                # sample measures its velocity before another model request.
                measure_held(other,packet)
                robot.hold(.10)
                packet=observe()
                candidates=[o for o in scene.objects if o.posture=='UPRIGHT' and not o.failures
                    and np.allclose(o.size_m,trackers[a].size,rtol=0,atol=.002)
                    and np.linalg.norm(np.subtract(o.position_m[:2],m['position_m'][:2]))<.035]
                if len(candidates)!=1 or resample is None:raise RuntimeError('REGRASP_TARGET_UNCONFIRMED')
                obj=candidates[0]
                alt=resample(a,obj,scene,packet,selected[a],goals[a])
                objects[a]=obj
                references[a]=PreCloseTarget(obj,scene,priors)
                selected[a]=asdict(alt)
                pair['selection'][arms.index(a)]=selected[a]
                recovery['replacement']=alt.candidate_id
                pos=np.asarray(alt.position_m)
                rot=pose_matrix(pos,alt.orientation_wxyz)[:3,:3]
                pre=pos-np.asarray(alt.approach_direction)*alt.pregrasp_distance_m
                move('regrasp_pregrasp',{a:(pre,rot)},active=(a,),carrying=(other,))
                packet=observe()
                proof=confirm_target(a,packet)
                if proof['full_pose_measured']:
                    scene=reconcile_preclose_target(estimator,scene,proof)
                    references[a]=references[a].renewed(scene,proof)
                    robot.set_observation(packet,scene,priors)
                move('regrasp_approach',{a:(pos,rot)},True,active=(a,),carrying=(other,))
                packet=observe()
                confirm_target(a,packet)
                fingers(True,(a,))
                packet=observe()
                proof=confirm_target(a,packet,closed=True)
                scene=reconcile_preclose_target(estimator,scene,proof)
                objects[a]=references[a].obj
                trackers[a]=CuboidHoldingTracker.from_preclose_support(objects[a],proof,*robot.fk(a),priors,packet=packet)
                release_trackers[a]=HeldSurfaceTracker.from_preclose_support(proof,*robot.fk(a),packet.observation_time_s)
                for b in arms:measure_held(b,packet)
                tcp,rot=robot.fk(a)
                closed_tcp_z[a]=tcp[2]
                closed_states[a]=robot.read_states()[a]
                raised[a]=(tcp+[0,0,.18],rot)
                move('regrasp_initial_lift',{a:(tcp+[0,0,.06],rot)},True,active=(a,))
                packet=observe()
                for b in arms:measure_held(b,packet)
                recovery['recheck']=slip(a)
                if recovery['recheck']['suspect']:raise RuntimeError('REGRASP_SLIP_REPEATED')
                recovery['success']=True
            finally:robot.stationary_payload_keepout=None
    robot.set_observation(packet,scene,priors)
    move('lift',raised,True)
    packet = observe()
    places = {}
    for a in arms:
        t = trackers[a]
        support = measure_held(a,packet)
        gain=support.get('height_gain_lower_bound_m') if support.get('grasp_constrained') else support['height_gain_m']
        if gain is None or gain < .08:
            raise RuntimeError('PAIR_LIFT_UNCONFIRMED:'+a)
        phases.setdefault('lift_evidence',{})[a]=support
        places[a] = placement_goal(goals[a],robot.place_support_z,t.size,.012,
                                   robot.fk(a)[1]@t.relative_rotation,t.offset,t.relative_rotation)
    robot.set_observation(packet,scene,priors)
    report['simultaneous_grasp_verified']=phases['lift']['overlap_ticks']>0
    robot.anchor_observation()
    before_transfer={xy:robot.observed_workspace.slot(xy) for xy in goals.values()}
    # Carry without simultaneously rotating the carton. Aim the BOX centre
    # above its lane using the measured grasp offset before changing attitude.
    overhead={}
    for a in arms:
        t=trackers[a]
        rot=robot.fk(a)[1]
        centre=np.array([*goals[a],robot.place_support_z+t.size[2]/2+.012+.18])
        overhead[a]=(centre-rot@t.offset,rot)
    move('transfer',overhead)
    packet=observe_destinations()
    for a in arms:
        observed=measure_held(a,packet)
        t=trackers[a]
        places[a]=placement_goal(goals[a],robot.place_support_z,t.size,.012,
                                 observed['rotation'],t.offset,t.relative_rotation)
    verify_slots(packet)
    robot.set_observation(packet,scene,priors)
    move('align',{a:(p+[0.,0.,.18],r) for a,(p,r) in places.items()})
    packet=observe_destinations()
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
        report.setdefault('release_check',{})[a]=dict(centre_m=measurement['position_m'],
            tilt_deg=math.degrees(math.acos(np.clip(rotation[2,2],-1,1))),gap_upper_m=float(upper_gap),
            measurement=measurement['status'],waypoint_reacquisition=measurement.get('waypoint_reacquisition'))
        if upper_gap>.020:raise RuntimeError('PAIR_RELEASE_HEIGHT_UNSUPPORTED:'+a)
    report['last_phase']='release'
    fingers(False)
    held=False
    robot.held_visual_geometry=None
    packet=observe()
    for a in arms:
        m=measurements[a]
        from workstation.planning.selection import nearest_upright_slot_rotation
        fit=release_trackers[a].collision_estimate(packet,scene,*robot.fk(a),
            max_drop_m=max(0.,report['release_check'][a]['gap_upper_m']),
            reference_position_m=m['position_m'],reference_rotation=m['rotation'],allow_partial=True)
        if fit is None:
            # A tilted bottom edge may settle flat on release. The new
            # depth image must support that level orientation and the footprint.
            trial=nearest_upright_slot_rotation(np.asarray(m['rotation']))
            hint=np.asarray(m['position_m']).copy()
            hint[2]=robot.place_support_z+trackers[a].size[2]/2
            fit=release_trackers[a].collision_estimate(packet,scene,*robot.fk(a),
                reference_position_m=hint,reference_rotation=trial,allow_partial=True)
        if fit is None:raise RuntimeError('PAIR_RELEASED_GEOMETRY_UNCERTAIN:'+a)
        tid=objects[a].track_id
        scene=estimator.associate_supported_target(scene,tid,fit)
        obj=next((o for o in scene.objects if o.track_id==tid),None)
        if obj is None:raise RuntimeError('PAIR_RELEASED_TRACK_UNRESOLVED:'+a)
        dimensions=fit.get('collision_size_m',trackers[a].size)
        bounded=replace(obj,position_m=fit['position_m'],axis_yaw_rad=fit['axis_yaw_rad'],
            size_m=tuple(trackers[a].size),uncertainty_m=fit['uncertainty_m'],measured_cuboid=None,
            posture='UNKNOWN',hypotheses=(),visibility='RELEASE_BOUNDED_CUBOID',failures=(),
            quality={**obj.quality,'geometric_length_m':dimensions[0],'geometric_width_m':dimensions[1],
                     'observed_height_m':dimensions[2]})
        scene=replace(scene,objects=tuple(bounded if o.track_id==tid else o for o in scene.objects))
        report.setdefault('released_bounds',{})[a]=fit
    robot.set_observation(packet,scene,priors)
    move('retreat',{a:(p+[0.,0.,.18],r) for a,(p,r) in places.items()},True)
    # Clear the two camera sightlines. The last measured release envelopes
    # remain stationary obstacles while the empty arms leave the conveyor.
    report['last_phase']='clear_for_verification'
    for a in arms:
        robot.execute_joint(a,robot.home,scene,None)
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
            if len(matches)==1 and slots.placement_observed(a,xy,matches[0].track_id):
                o,=matches
                placements[a]=dict(initial_track=selected[a]['target_id'],grasp_track=objects[a].track_id,observed_track=o.track_id,
                    goal_xy=xy,visual_error_xy_m=(np.asarray(o.position_m[:2])-xy).tolist())
        ids=tuple(p['observed_track'] for p in placements.values())
        current=stationary_slot_tracks(robot.observed_workspace,scene,unused)
        ready=len(set(ids))==2 and parked_unchanged(current)
        if not ready:
            stable_since=None
        elif stable_since is None or ids!=previous_ids:
            stable_since=scene.observation_time_s
        previous_ids=ids
        if ready and scene.observation_time_s-stable_since>=.5:
            for a,p in placements.items():
                slots.confirm_placement(a,goals[a],p['observed_track'])
            report['placements']=placements
            report['parked_after_placement']=[dict(xy=list(k),track_id=o.track_id,position_m=list(o.position_m),
                centre_change_m=float(np.linalg.norm(np.subtract(o.position_m,parked[k]['position_m']))))
                for k,o in current.items()]
            break
    else:
        from workstation.diagnostics.vision_check import save_observation
        packet=observe()
        save_observation(env.output,'pair_final_verification',packet,scene)
        raise RuntimeError('PAIR_RELEASED_CARTONS_NOT_STABLE')
    report['slots_final']=slots.report()
