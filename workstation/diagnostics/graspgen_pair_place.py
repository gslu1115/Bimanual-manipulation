"""Two independently grasped cartons, joint commands on a shared physics clock."""
import math
from time import perf_counter
from dataclasses import replace, asdict
import numpy as np

from workstation.observations.camera_geometry import pose_matrix, quaternion_from_matrix
from workstation.perception.held_surface import CuboidHoldingTracker, HeldSurfaceTracker
from workstation.perception.preclose_target import PreCloseTarget, preclose_supported, reconcile_preclose_target
from workstation.perception.scene_estimate import ScenePriors, pose_is_observed
from workstation.planning.collision import AABB, sphere_box_clearance, check_spheres, carton_contact_allowed
from workstation.planning.graspgen_candidates import revalidate_candidate
from workstation.skills.cumotion_pick_place import placement_goal
from workstation.diagnostics.graspgen_pair import drive_empty_pregrasps, synchronized_knots


class PairClearanceError(RuntimeError):
    def __init__(self, reason, parts, clearance, required):
        super().__init__(reason)
        self.details=dict(parts=parts,clearance_m=float(clearance),required_m=required)


def transfer_schedule(robot,paths,payload_guard,*,rejections=None):
    """Choose a clear timing of the same two verified spatial paths before moving."""
    arms=('panda_left','panda_right')
    for delay,waiting in [(0.,0),(.35,0),(.35,1),(.7,0),(.7,1),(1.,0),(1.,1)]:
        scheduled=list(paths)
        if delay:
            for i,path in enumerate(paths):
                wait=[path[0] if i==waiting else path[-1]]*math.ceil((len(path)-1)*delay)
                scheduled[i]=wait+list(path) if i==waiting else list(path)+wait
        detail=dict(delayed_arm=arms[waiting] if delay else None,delay_fraction=delay)
        rejection={} if rejections is not None else None
        try:
            for index,(left,right) in enumerate(synchronized_knots(*scheduled,max_step=.015)):
                reason,_=check_spheres(robot.spheres(arms[0],left),robot.spheres(arms[1],right),[],None,
                                       margin=.008,rejection=rejection)
                if reason:raise RuntimeError(reason)
                payload_guard(left,right)
        except RuntimeError as error:
            if rejections is not None:
                if 'link_indices' in rejection:
                    owners=arms if str(error)=='OTHER_ARM_COLLISION' else (arms[0],arms[0])
                    rejection['parts']=[dict(arm=arm,link_index=link,sphere_index=sphere)
                        for arm,link,sphere in zip(owners,rejection['link_indices'],rejection['sphere_indices'])]
                rejections.append(dict(detail,reason=str(error),sample_index=index,
                    joint_positions_rad={arms[0]:left.tolist(),arms[1]:right.tolist()},
                    **(error.details if isinstance(error,PairClearanceError) else rejection)))
            continue
        return scheduled,detail
    raise RuntimeError('NO_CLEAR_PAIRED_TRANSFER_SCHEDULE')


def path_rejection(reason):
    return reason.startswith(('IK_','JOINT_LIMIT','NO_CLEAR_','OTHER_ARM_COLLISION','SELF_COLLISION',
                              'STATIC_PATH_BLOCKED','OBSERVED_PATH_OBSTACLE','HELD_CARTON_',
                              'PAYLOAD_','PARKED_CARTON_CLEARANCE'))


def choose_transfer_strategy(policy,planners,attempts):
    """Try deterministic feasibility plans only. Execution errors never enter here."""
    if policy not in ('auto','orient-first','sequential-end'):raise ValueError('Unknown pair transfer policy')
    order=('orient-first','sequential-end') if policy=='auto' else (policy,)
    for name in order:
        start=perf_counter()
        try:plan=planners[name]()
        except RuntimeError as error:
            # Missing perception/holding evidence must stop, not trigger motion
            # through a different plan with the same invalid state estimate.
            reason=str(error)
            attempts.append(dict(strategy=name,feasible=False,reason=reason,planning_wall_s=perf_counter()-start))
            if not path_rejection(reason):raise
            continue
        attempts.append(dict(strategy=name,feasible=True,planning_wall_s=perf_counter()-start))
        return name,plan
    raise RuntimeError('NO_CLEAR_PAIRED_PLACEMENT_STRATEGY')


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


def supported_payload_fragments(packet,scene,measurements,sizes):
    """Associate failed fragments only to a unique CURRENT measured cuboid surface."""
    from workstation.perception.rgbd_cartons import deproject
    clouds={};joined={a:[] for a in measurements}
    used={ref for m in measurements.values() for ref in m['mask_refs']}
    for obj in scene.objects:
        if not obj.failures or not obj.mask_refs or used.intersection(obj.mask_refs):continue
        candidates=[]
        for arm,m in measurements.items():
            if m['time_s']!=packet.observation_time_s:continue
            ratios=[]
            for ref in obj.mask_refs:
                name=ref.split('/')[0];frame=packet.cameras.get(name);mask=scene.masks.get(ref)
                if (frame is None or packet.camera_status.get(name)!='OK' or mask is None or
                        frame.sample_time_s!=packet.observation_time_s or mask.shape!=frame.depth_m.shape or
                        not mask.any() or frame.valid_depth[mask].mean()<.9):break
                if name not in clouds:clouds[name]=deproject(frame)
                points=clouds[name][mask&frame.valid_depth]
                if len(points)<60:break
                local=(points-m['position_m'])@np.asarray(m['rotation'])
                half=np.asarray(sizes[arm])/2
                inside=np.all(np.abs(local)<=half+.006,axis=1)
                surface=np.min(np.abs(np.abs(local)-half),axis=1)<=.006
                ratio=float((inside&surface).mean())
                if ratio<.95:break
                ratios.append(ratio)
            else:
                if ratios:candidates.append((arm,min(ratios)))
        if len(candidates)==1:
            arm,ratio=candidates[0]
            joined[arm].append(dict(track_id=obj.track_id,mask_refs=obj.mask_refs,surface_ratio=ratio))
    return joined


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
    depth_ref=None
    if fit['status']=='UNKNOWN':
        extra=parked_depth_top(packet,reference,masks)
        if extra is not None:
            ref,mask=extra
            extended=replace(scene,masks={**scene.masks,ref:mask})
            measured=fit_cuboid_pose(packet,extended,reference['position_m'],prior_rotation,size,priors)
            if measured_cuboid_supported(measured,packet.observation_time_s) and set(measured['mask_refs'])==set(masks)|{ref}:
                fit=measured;depth_ref=ref
    full=measured_cuboid_supported(fit,packet.observation_time_s)
    if full or reference_constrained_cuboid_supported(fit,packet.observation_time_s):
        if set(fit['mask_refs'])!=set(masks)|({depth_ref} if depth_ref else set()):return None
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
        time_s=packet.observation_time_s,mask_refs=masks,additional_depth_component=depth_ref,
        observed_face_axes=face_axes,bounded_axis=axis,offset_interval_m=interval,
        collision_center_m=centre.tolist(),collision_size_m=world.tolist())


def parked_depth_top(packet,reference,instance_refs,*,allow_partial=False):
    """Use an uncut depth component; a clipped top can corroborate an existing keepout."""
    import cv2
    from workstation.perception.rgbd_cartons import deproject
    centre=np.asarray(reference['position_m']);size=np.asarray(reference['size_m'])
    yaw=reference['axis_yaw_rad'];c,s=math.cos(yaw),math.sin(yaw)
    basis=np.array([[c,-s],[s,c]])
    used={ref.split('/')[0] for ref in instance_refs}
    for name,frame in packet.cameras.items():
        if name in used or frame is None or packet.camera_status.get(name)!='OK' or frame.sample_time_s!=packet.observation_time_s:continue
        xyz=deproject(frame);top=frame.valid_depth & (np.abs(xyz[:,:,2]-centre[2]-size[2]/2)<.0015)
        count,labels,stats,_=cv2.connectedComponentsWithStats(top.astype(np.uint8),connectivity=8)
        for i in range(1,count):
            if stats[i,cv2.CC_STAT_AREA]<100:continue
            mask=labels==i;local=(xyz[mask,:2]-centre[:2])@basis
            bounds=np.quantile(local,[.002,.998],axis=0)
            errors=np.abs(bounds-[-size[:2]/2,size[:2]/2])
            if allow_partial:
                if (np.mean(np.all(np.abs(local)<=size[:2]/2+.004,axis=1))<.99 or
                        np.min(np.max(errors,axis=0))>.004 or
                        np.min((bounds[1]-bounds[0])/size[:2])<.65):continue
            elif np.max(errors)>.004:continue
            coverage=cv2.contourArea(cv2.convexHull(local.astype(np.float32)))/np.prod(size[:2])
            if not (.65 if allow_partial else .8)<=coverage<=1.12:continue
            return f'{name}/{frame.sequence_id}/depth_plane',mask
    return None


def parked_side_with_top(packet,scene,reference,refs,priors,*,guarded=False):
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
        if not guarded and not parked_top_still_visible(packet,reference,exclude_camera=name):continue
        return centre+direction*(level-np.sign(level)*size[axis]/2),rotation,1-axis,margin
    return None


def parked_top_still_visible(packet,reference,mask_refs=(),masks=None,*,exclude_camera=None,
                             allow_cross_view_top=False):
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
        if selected and name not in selected and not allow_cross_view_top:continue
        if f is None or packet.camera_status.get(name)!='OK' or f.sample_time_s!=packet.observation_time_s:
            if name in selected:return False
            continue
        xyz=deproject(f)
        local=(xyz[:,:,:2]-centre[:2])@rz
        mask=(f.valid_depth & np.all(np.abs(local)<=size[:2]/2,axis=2)
              & (np.abs(xyz[:,:,2]-centre[2]-size[2]/2)<.001))
        if name in selected:
            instance=masks[selected[name]]
            if not instance.any() or f.valid_depth[instance].mean()<.9:return False
            observed=np.column_stack((local[instance&f.valid_depth],xyz[:,:,2][instance&f.valid_depth]-centre[2]))
            residual=np.abs(observed)-size/2
            if np.mean((residual.max(axis=1)<=.004)&(np.abs(residual).min(axis=1)<=.004))<.95:return False
            if not allow_cross_view_top:mask&=instance
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
        for index in range(1,count):
            if stats[index,cv2.CC_STAT_AREA]<9:continue
            patch=xyz[labels==index]
            if np.ptp(np.quantile(patch[:,2],[.05,.95]))<.001 and np.min(np.ptp(patch[:,:2],axis=0))>.003:
                confirmed=True
    return confirmed


def parked_top_occluded(packet,reference,*,allow_wrist_out_of_view=False):
    """All valid rays into the inner top footprint must terminate in front of it."""
    u,v=np.meshgrid(np.linspace(-.42,.42,11),np.linspace(-.42,.42,9))
    size=np.asarray(reference['size_m']);yaw=reference['axis_yaw_rad']
    c,s=math.cos(yaw),math.sin(yaw)
    local=np.column_stack((u.ravel()*size[0],v.ravel()*size[1]))
    world=np.tile(reference['position_m'],(u.size,1))
    world[:,:2]+=local@np.array([[c,s],[-s,c]])
    world[:,2]+=size[2]/2
    views=0
    global_covered=False
    if allow_wrist_out_of_view and set(packet.cameras)!={'scene_camera','left_wrist_camera','right_wrist_camera'}:return False
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
        if name=='scene_camera' and inside.all():global_covered=True
    return views>=2 or (allow_wrist_out_of_view and global_covered)


def place_pair(env, robot, estimator, scene, pair, slots, report,alternatives=None,resample=None):
    arms = ('panda_left', 'panda_right')
    priors = ScenePriors.from_config(env.c)
    selected = dict(zip(arms, pair['selection']))
    goals = dict(zip(arms, pair['slots']))
    unused=[xy for xy in slots.positions if xy not in goals.values()]
    parked={tuple(p['xy']):p for p in report.get('parked_before_pick',[])}
    initially_occupied={tuple(s['xy']) for s in report['slots_initial'] if s['state']=='occupied'}
    partial_keepout={}
    transfer_top_seen={}
    objects = {a: next(o for o in scene.objects if o.track_id == selected[a]['target_id']) for a in arms}
    from workstation.diagnostics.graspgen_pair import source_cartons
    from workstation.planning.pair_task import match_stationary_sources
    source_references={o.track_id:dict(position_m=o.position_m,size_m=o.size_m,axis_yaw_rad=o.axis_yaw_rad)
        for o in source_cartons(scene,env.c['table']) if o.track_id not in {v.track_id for v in objects.values()}}
    source_keepouts=[]
    for key,ref in source_references.items():
        c,s=abs(math.cos(ref['axis_yaw_rad'])),abs(math.sin(ref['axis_yaw_rad']))
        sx,sy,sz=ref['size_m']
        source_keepouts.append((AABB('unselected_'+key,np.asarray(ref['position_m']),
            np.array([c*sx+s*sy,s*sx+c*sy,sz])+.024,'carton'),scene.observation_time_s))
    report['unselected_source_references']=source_references
    report['contact_policy']=dict(carton_carton='allowed; geometric gaps recorded, not contact forces',
        robot_self='protected',robot_robot='protected',robot_carton='protected',fixed_station='protected')
    for obj in objects.values():estimator.mark_identity(obj.identity_id,'selected')
    trackers = {}
    release_trackers = {}
    measurements = {}
    held = False
    carried_arms=set(arms)
    released_references={}
    released_keepout={}
    sequential=False
    fresh_scene=scene
    before_transfer={}
    phases = report['phases'] = {}

    def observe():
        nonlocal scene,fresh_scene
        packet = env.get_observation_packet(refresh=True)
        scene = estimator.estimate(packet)
        fresh_scene=scene
        if released_references:refresh_released(packet)
        robot.set_observation(packet, scene, priors)
        slots.observe(robot.observed_workspace)
        return packet

    def confirm_target(arm,packet,closed=False):
        nonlocal scene
        from workstation.diagnostics.vision_check import save_observation
        prior=references[arm]
        renewed,proof=prior.evaluate(packet,scene,closed=closed)
        report.setdefault('preclose_evidence',[]).append(dict(proof,arm=arm))
        if not preclose_supported(proof,packet.observation_time_s):
            report['next_action']=proof['next_action'] if not closed else 'STOP'
            report['perception_failure']=dict(arm=arm,reason=proof['reason'],phase=report['last_phase'])
            save_observation(env.output,'pair_'+arm+'_'+report['last_phase'],packet,scene)
            raise RuntimeError('PAIR_TARGET_UNSUPPORTED:'+arm+':'+proof['reason'])
        scene=reconcile_preclose_target(estimator,scene,proof)
        references[arm]=renewed
        if renewed is not prior:
            # Complete current geometry plus retained identity/orientation.
            obj=renewed.obj
            objects[arm]=obj
            scene=replace(scene,objects=tuple(obj if o.track_id==obj.track_id else o for o in scene.objects))
            estimator._tracks[obj.track_id]=obj
        report['next_action']='CONTINUE'
        return proof

    def carried_clearance(left, right):
        boxes = []
        for arm, q, other, other_q in zip(arms, (left, right), reversed(arms), (right, left)):
            if arm not in carried_arms:continue
            pos, rot = robot.fk(arm, q)
            t = trackers[arm]
            centre = pos + rot @ t.offset
            extent = np.abs(rot @ t.relative_rotation) @ (t.size + 2*t.uncertainty_m)
            box = AABB(objects[arm].track_id, centre, extent,'carton')
            payload=dict(arm=arm,track_id=box.name)
            keepouts=list(partial_keepout.values())+list(released_keepout.values())+source_keepouts
            for keepout,_ in keepouts:
                payload_gap=float(np.max(np.abs(centre-keepout.centre)-(extent+keepout.size)/2))
                robot_gap,sphere,link=min((sphere_box_clearance(p,r,keepout),index,link)
                    for index,(link,p,r) in enumerate(robot.spheres(arm,q)))
                key=('unselected_clearances' if keepout.name.startswith('unselected_') else
                     'released_clearances' if keepout.name.startswith('released_') else 'parked_clearances')
                stats=report.setdefault(key,{}).setdefault(arm,{})
                for label,gap in [('carton_axis_gap_m',payload_gap),('arm_surface_gap_m',robot_gap)]:
                    stats[label]=min(stats.get(label,gap),float(gap))
                if payload_gap<=.008 and not carton_contact_allowed(box,keepout):
                    raise PairClearanceError('PARKED_CARTON_CLEARANCE', [payload,dict(obstacle=keepout.name)],
                                             payload_gap,.008)
                if robot_gap<.008:
                    raise PairClearanceError('PARKED_CARTON_CLEARANCE',
                        [dict(arm=arm,link_index=link,sphere_index=sphere),dict(obstacle=keepout.name)],robot_gap,.008)
            for index,(link,c,r) in enumerate(robot.spheres(other,other_q)):
                gap=sphere_box_clearance(c,r,box)
                if gap<.002:
                    raise PairClearanceError('PAYLOAD_OTHER_ARM_COLLISION',
                        [payload,dict(arm=other,link_index=link,sphere_index=index)],gap,.002)
            boxes.append((box,payload))
        if len(boxes)!=2:return
        (a,part_a),(b,part_b)=boxes
        gap=float(np.max(np.abs(a.centre-b.centre)-(a.size+b.size)/2))
        report['payload_pair_min_axis_gap_m']=min(report.get('payload_pair_min_axis_gap_m',gap),gap)
        if gap<=.002 and not carton_contact_allowed(a,b):
            raise PairClearanceError('PAYLOAD_PAYLOAD_COLLISION',[part_a,part_b],gap,.002)

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
                all(matches_parked_reference(k,o) for k,o in current.items()))

    def matches_parked_reference(xy,obj):
        return (xy in parked and np.linalg.norm(np.subtract(obj.position_m,parked[xy]['position_m']))<=.005
                and np.allclose(obj.size_m,parked[xy]['size_m'],atol=.002,rtol=0))

    def protect_stationary(current,time_s):
        if env.conveyor.state!='WAITING' or env.safe_to_index or not parked_unchanged(current):return
        for xy,o in current.items():
            c,s=math.cos(o.axis_yaw_rad),math.sin(o.axis_yaw_rad)
            R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            box=AABB('parked_carton',np.asarray(o.position_m),np.abs(R)@np.asarray(o.size_m)+.016,'carton')
            partial_keepout[xy]=(box,time_s)
            robot.stationary_payload_keepout=box
            report.setdefault('parked_protection',[]).append(dict(phase=report['last_phase'],time_s=time_s,
                xy=list(xy),track_id=o.track_id,centre_m=list(o.position_m),size_m=box.size.tolist()))

    def verify_slots(packet,active=arms):
        nonlocal scene
        # Reconcile BOTH carried cartons before testing the slot underneath
        # either one. Register the current arm-specific payload to explain its
        # shadow using saved free depth rays, never using ownership as clearance.
        observed=stationary_slot_tracks(robot.observed_workspace,scene,unused)
        current={xy:o for xy,o in observed.items() if matches_parked_reference(xy,o)}
        parked_ids={o.track_id for o in current.values()}
        carried_ids={objects[a].track_id for a in carried_arms}
        released_ids={objects[a].track_id for a in released_references}
        enclosed=set()
        if env.conveyor.state=='WAITING' and not env.safe_to_index:
            protect_stationary(current,packet.observation_time_s)
            for xy in parked.keys()-current.keys():
                ids=robot.observed_workspace.slot(xy)['observed_objects']
                fragments=[o for o in scene.objects if o.track_id in ids]
                if carried_ids.intersection(ids):continue
                if xy in partial_keepout:
                    box,seen=partial_keepout[xy]
                    refs=[ref for obj in fragments for ref in obj.mask_refs]
                    age=packet.observation_time_s-seen
                    # A currently matched whole instance corroborates the protected
                    # obstacle throughout the same 20 s reserved-slot inspection.
                    # Beyond 8 s an unsegmented patch also needs foreground
                    # occlusion across the sampled interior of the guarded box.
                    shadow=not ids and 0<age<=20. and parked_top_occluded(packet,parked[xy])
                    patch=(0<age<=(20. if refs or shadow else 8.) and
                           parked_top_still_visible(packet,parked[xy],refs,scene.masks,
                                                    allow_cross_view_top=True))
                    component=None
                    if not patch and not ids and 0<age<=20.:
                        component=parked_depth_top(packet,parked[xy],(),allow_partial=True)
                    recent_transfer_top=(report['last_phase']=='align' and xy in transfer_top_seen
                        and seen<=transfer_top_seen[xy]<packet.observation_time_s)
                    occluded=shadow and not patch and (age<=8. or recent_transfer_top)
                    side_face=(bool(refs) and not patch and 0<age<=20. and
                        parked_side_with_top(packet,scene,parked[xy],refs,priors,guarded=True) is not None)
                    # During a protected transfer both wrists can look away.
                    # Require the global view to witness occlusion of the full
                    # grid and reject any contrary depth in either wrist view.
                    lift_proof=next((p for p in reversed(report.get('parked_protection',[]))
                                     if tuple(p['xy'])==xy and p['phase']=='lift'),None)
                    if (not ids and not patch and component is None and not occluded and lift_proof
                            and 0<packet.observation_time_s-lift_proof['time_s']<=20.):
                        occluded=parked_top_occluded(packet,parked[xy],allow_wrist_out_of_view=True)
                    # A catalogue/centre conflict on a nominally complete observation
                    # requires the explicit cuboid fit below, even with a visible top.
                    if xy not in observed and (patch or side_face or component is not None or occluded):
                        if component is not None and report['last_phase']=='transfer':
                            transfer_top_seen[xy]=packet.observation_time_s
                        enclosed.add(xy)
                        if ids and (patch or side_face):
                            # Every complete contributing mask was checked against
                            # the protected carton. Keep one obstacle at its guarded
                            # bounds instead of placing another cube at a face centroid.
                            obj=fragments[0]
                            proxy=replace(obj,position_m=tuple(box.centre),axis_yaw_rad=0.,
                                size_m=tuple(box.size),mask_refs=tuple(refs),posture='UNKNOWN',
                                visibility='PARKED_COLLISION_ENVELOPE',failures=(),hypotheses=(),
                                measured_cuboid=None,quality={**obj.quality,'geometric_length_m':box.size[0],
                                    'geometric_width_m':box.size[1],'observed_height_m':box.size[2]})
                            scene=replace(scene,objects=tuple(o for o in scene.objects if o.track_id not in ids)+(proxy,))
                            parked_ids.add(proxy.track_id)
                        else:parked_ids.update(ids)
                        report.setdefault('parked_occlusion_checks',[]).append(dict(
                            xy=list(xy),mask_refs=refs,time_s=packet.observation_time_s,last_surface_s=seen,
                            depth_component=component[0] if component is not None else None,
                            prior_transfer_top_s=transfer_top_seen.get(xy) if occluded else None,
                            source=('current signed side plane' if side_face else 'current top-depth patch'
                                    if patch or component is not None else 'foreground occlusion in current depth views')
                                   +' and continuously protected collision envelope'))
                        continue
                if not fragments:continue
                bound=parked_surface_envelope(packet,scene,parked[xy],fragments,priors,goals.values())
                if bound is None:continue
                proxy,proof=bound
                box=AABB('parked_carton',np.asarray(proxy.position_m),np.asarray(proxy.size_m),'carton')
                partial_keepout[xy]=(box,packet.observation_time_s)
                robot.stationary_payload_keepout=box
                scene=replace(scene,objects=tuple(o for o in scene.objects if o.track_id not in ids)+(proxy,))
                report.setdefault('parked_surface_bounds',[]).append(proof)
                parked_ids.add(proxy.track_id);enclosed.add(xy)
        supported=parked_unchanged(current,enclosed)
        current_sources=match_stationary_sources(scene,source_references)
        source_ids={o.track_id for o in current_sources.values()}
        sources_supported=current_sources.keys()==source_references.keys()
        report.setdefault('unselected_source_checks',[]).append(dict(time_s=packet.observation_time_s,
            supported=sources_supported,observed_tracks=sorted(source_ids)))
        for arm in active:
            t=trackers[arm]
            robot.set_observation(packet,scene,priors)
            robot.held_visual_geometry=dict(arm=arm,target_id=objects[arm].track_id,offset=t.offset,
                relative_rotation=t.relative_rotation,uncertainty_m=t.uncertainty_m,allow_grasp_translation=True)
            if not robot.confirm_payload_observation(objects[arm].track_id,t,measurements[arm],packet):
                raise RuntimeError('PAIR_PAYLOAD_GEOMETRY_UNCONFIRMED:'+arm)
            xy=goals[arm]
            view=robot.observed_workspace.slot(xy)
            if sequential and view['free']:before_transfer[xy]=view
            accounted=(len(carried_ids|released_ids)==2 and supported and carried_ids.isdisjoint(parked_ids) and
                       all(o.track_id in carried_ids|released_ids|parked_ids or
                           np.linalg.norm(np.asarray(o.position_m[:2])-xy) >
                           .15+np.linalg.norm(o.size_m)/2+o.uncertainty_m for o in scene.objects) and
                       all(report['released_bounds'][a]['time_s']==packet.observation_time_s for a in released_references) and
                       all(measurements[a]['time_s']==packet.observation_time_s and
                           measurements[a]['position_m'][2]-np.linalg.norm(trackers[a].size)/2>
                           robot.place_support_z+.04 for a in carried_arms))
            evidence=slots.approach_clearance(arm,xy,view,before_transfer.get(xy),accounted_for=accounted,
                stationary_belt=env.conveyor.state=='WAITING' and not env.safe_to_index)
            if evidence=='observed_empty_before_transfer_both_cartons_in_hand' and released_ids:
                evidence='observed_empty_before_transfer_current_held_and_released_geometry'
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
            proof=confirm_target(arm,packet)
            report.setdefault('approach_reference_checks',[]).append(dict(
                arm=arm,time_s=packet.observation_time_s,reason=proof['reason'],
                renewed=references[arm] is not prior))
        robot.set_observation(packet,scene,priors)

    from workstation.planning.plan_reuse import PreviewPlans,array_key,encoder_key
    previews=PreviewPlans()

    def move(phase, destinations, contact=False,*,active=arms,carrying=None,
             dry_run=False,start_states=None,label=None):
        nonlocal next_approach_check
        report['last_phase'] = phase
        label=label or phase
        report['phase_checkpoint']=dict(phase=phase,event='planning',dry_run=dry_run)
        env._write_json('graspgen_pair_check.json',report)
        print('[pair-phase] '+phase+' planning',flush=True)
        start_wall=perf_counter()
        carrying=(set(arms) if held else set()) if carrying is None else set(carrying)
        # Reuse only an exactly unchanged live context. In particular the
        # post-orientation payload refresh invalidates the translation preview.
        # Cache misses use this same planner, not another recovery branch.
        context=(phase,contact,held,tuple(active),tuple(sorted(carrying)),
            id(packet),packet.observation_time_s,id(scene),encoder_key(robot.read_states()),
            None if start_states is None else tuple((a,array_key(q)) for a,q in sorted(start_states.items())),
            tuple((a,array_key(p),array_key(r)) for a,(p,r) in sorted(destinations.items())),
            tuple((a,array_key(trackers[a].offset),array_key(trackers[a].relative_rotation),
                   trackers[a].uncertainty_m) for a in sorted(carrying)),
            repr(measurements),repr(getattr(robot,'stationary_payload_keepout',None)),sequential)
        reused=None if dry_run else previews.get(phase,context)
        if reused is not None:
            paths,schedule=reused
            if schedule is not None:report[label+'_schedule']=schedule
        else:
            schedule=None
            paths = []
            for arm in arms:
                o = objects[arm]
                q = (robot.read_states()[arm].joint_positions_rad if start_states is None else start_states[arm])
                if arm not in active:
                    paths.append([q,q])
                    continue
                carries=arm in carrying
                pos, rot = robot.fk(arm, q)
                goal, goal_rot = destinations[arm]
                try:
                    path = robot.cartesian_knots(arm, pos, goal, 0., 0., .12, q,
                        orientation0=quaternion_from_matrix(rot), orientation1=quaternion_from_matrix(goal_rot))
                except RuntimeError as exc:
                    if phase!='approach' or str(exc)!='IK_DISCONTINUITY':raise
                    # Neither arm has started the descent. Check a nearby alternate
                    # from the actual pregrasp FK against the current target.
                    seq={n:f.sequence_id for n,f in packet.cameras.items() if f}
                    for alt in (alternatives or {}).get(arm,()):
                        if alt.candidate_id==selected[arm]['candidate_id']:continue
                        try:
                            alt=revalidate_candidate(replace(alt,target_id=o.track_id),o,scene,seq)
                            # Pending backups enter the same current-frame
                            # prefix/descent checks below. Later lift/transfer
                            # paths are checked before their own execution;
                            # a prefix score never authorizes a future motion.
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
                        check_local_unknown=not held and phase=='approach',check_approach_corridor=phase=='approach',
                        other_joint_positions_rad=None if start_states is None else start_states[next(a for a in arms if a!=arm)])
                except RuntimeError:
                    from workstation.diagnostics.vision_check import save_observation
                    if not dry_run:save_observation(env.output,'pair_blocked_'+phase,packet,scene)
                    report['next_action']='REPLAN' if phase=='approach' and not report.get('empty_approach_started') else 'STOP'
                    report['path_failure']=dict(arm=arm, details=robot.last_plan_check,
                        robot_pixels_removed=estimator.last_robot_filter_counts.copy())
                    raise
                paths.append(path)
            # Final Cartesian contact is verified below in metres/radians. The
            # per-step 0.04 rad stop and actual two-arm collision checks stay active.
            transfer=phase in ('transfer','align','transfer_orient','transfer_after_orient')
            if transfer or sequential:
                rejections=report.setdefault(label+'_schedule_rejections',[])
                try:
                    paths,schedule=transfer_schedule(robot,paths,carried_clearance,rejections=rejections)
                except RuntimeError:
                    report[label+'_rejected_paths_rad']={arm:np.asarray(path).tolist() for arm,path in zip(arms,paths)}
                    raise
                report[label+'_schedule']=schedule

        if dry_run:previews.put(phase,context,paths,schedule)
        report.setdefault('preview_path_reuse',[]).append(dict(phase=phase,dry_run=dry_run,
            reused=reused is not None,scope='exact live observation, encoders, payload and goal; guards retained'))
        transfer=phase in ('transfer','align','transfer_orient','transfer_after_orient')
        report.setdefault('motion_planning_wall_s',{})[label]=perf_counter()-start_wall
        if dry_run:return {a:np.asarray(path[-1]) for a,path in zip(arms,paths)}
        both_loaded=held and carrying==set(arms)
        renew_reference=phase in ('approach','regrasp_pregrasp','regrasp_approach')
        if renew_reference:next_approach_check=3.
        def guard(left,right):
            if sequential:
                reason,_=check_spheres(robot.spheres(arms[0],left),robot.spheres(arms[1],right),[],None,margin=.008)
                if reason:raise RuntimeError('PAIR_SEQUENTIAL_'+reason)
            carried_clearance(left,right)
        execution_start=perf_counter()
        report['phase_checkpoint']=dict(phase=phase,event='executing',dry_run=False)
        env._write_json('graspgen_pair_check.json',report)
        if phase=='approach':report['empty_approach_started']=True
        phases[label] = drive_empty_pregrasps(robot, paths, guard if both_loaded or sequential else None,
                                              following_band=.035 if contact else .025,require_overlap=not transfer and len(active)==2,
                                              speed_fraction=.4 if transfer else 1.,loaded_transfer=transfer,
                                              progress_check=(lambda elapsed:refresh_approach_references(elapsed,active)) if renew_reference else None)
        phases[label]['execution_wall_s']=perf_counter()-execution_start
        report['phase_checkpoint']=dict(phase=phase,event='completed',dry_run=False)
        env._write_json('graspgen_pair_check.json',report)
        for arm,(goal,goal_rot) in destinations.items():
            pos,rot=robot.fk(arm)
            error=float(np.linalg.norm(pos-goal))
            angle=float(np.arccos(np.clip((np.trace(rot.T@goal_rot)-1.)/2.,-1.,1.)))
            phases[label].setdefault('tcp_errors',{})[arm]=dict(position_m=error,angle_rad=angle)
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
    # The same evidence contract is used at the waypoint, during approach,
    # and after closure. One bounded fresh observation precedes task replanning.
    for observation_attempt in range(2):
        try:
            for a in arms:confirm_target(a,packet)
        except RuntimeError:
            if report.get('next_action')!='OBSERVE' or observation_attempt:raise
            robot.hold(.1)
            packet=observe()
            report.setdefault('local_reobservations',[]).append(dict(arm=a,time_s=packet.observation_time_s))
        else:break
    report.pop('perception_failure',None)
    robot.set_observation(packet,scene,priors)
    next_approach_check=3.
    move('approach', poses, True)
    packet = observe()
    for a in arms:
        proof=confirm_target(a,packet)
    report['last_phase']='close'
    report['closure_started']=True
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
                np.abs(m['rotation'])@(trackers[other].size+2*m['uncertainty_m']),'carton')
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
    for obj in objects.values():estimator.mark_identity(obj.identity_id,'held')
    move('lift',raised,True)
    packet = observe_destinations()
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
    protect_stationary(stationary_slot_tracks(robot.observed_workspace,scene,unused),packet.observation_time_s)
    # Carry without simultaneously rotating the carton. Aim the BOX centre
    # above its lane using the measured grasp offset before changing attitude.
    overhead={}
    for a in arms:
        t=trackers[a]
        rot=robot.fk(a)[1]
        centre=np.array([*goals[a],robot.place_support_z+t.size[2]/2+.012+.18])
        overhead[a]=(centre-rot@t.offset,rot)

    def refresh_released(current_packet):
        nonlocal scene
        packet=current_packet
        for a,m in released_references.items():
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
            c,s=math.cos(fit['axis_yaw_rad']),math.sin(fit['axis_yaw_rad'])
            rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            released_keepout[a]=(AABB('released_'+a,np.asarray(fit['position_m']),
                np.abs(rotation)@np.asarray(dimensions)+2*fit['uncertainty_m'],'carton'),packet.observation_time_s)

    def refresh_payloads():
        nonlocal packet,scene
        packet=observe_destinations()
        for a in arms:
            if a not in carried_arms:continue
            measured=measure_held(a,packet)
            if sequential:
                gain=measured.get('height_gain_lower_bound_m') if measured.get('grasp_constrained') else measured.get('height_gain_m')
                if gain is None and measured.get('geometric_pose_measured'):
                    gain=measured['position_m'][2]-trackers[a].initial_z
                report.setdefault('sequential_hold_checks',[]).append(dict(arm=a,phase=report['last_phase'],
                    time_s=packet.observation_time_s,status=measured['status'],height_gain_lower_bound_m=gain,
                    position_m=measured['position_m'],currently_carried=sorted(carried_arms)))
                if gain is None or gain<.08:raise RuntimeError('PAIR_WAITING_HOLD_UNCONFIRMED:'+a)
        current={a:measurements[a] for a in arms if a in carried_arms}
        fragments=supported_payload_fragments(packet,scene,current,{a:trackers[a].size for a in current})
        for a,items in fragments.items():
            if not items:continue
            refs=tuple(dict.fromkeys(tuple(current[a]['mask_refs'])+tuple(ref for item in items for ref in item['mask_refs'])))
            scene=estimator.associate_supported_target(scene,objects[a].track_id,dict(current[a],mask_refs=refs))
            report.setdefault('held_fragment_associations',[]).append(dict(arm=a,time_s=packet.observation_time_s,fragments=items))
        robot.set_observation(packet,scene,priors)
        protect_stationary(stationary_slot_tracks(robot.observed_workspace,scene,unused),packet.observation_time_s)
        if sequential:
            for a in carried_arms:
                view=robot.observed_workspace.slot(goals[a])
                if view['free']:before_transfer[goals[a]]=view

    def update_places(active):
        for a in active:
            t=trackers[a]
            places[a]=placement_goal(goals[a],robot.place_support_z,t.size,.012,
                                     measurements[a]['rotation'],t.offset,t.relative_rotation)

    def reobserve_second_slot(a):
        from workstation.planning.task_targets import reobserve_reserved_slot
        from workstation.diagnostics.vision_check import save_observation
        if carried_arms!={a} or not report.get('sequential_first_placement'):
            raise RuntimeError('PAIR_SLOT_REOBSERVE_PHASE_UNSUPPORTED')
        entry=dict(arm=a,checks=[],success=False,prior=before_transfer.get(goals[a]),
                   scope='new depth before second placement; expiry remains 20 seconds')
        report.setdefault('slot_reobservations',[]).append(entry)
        start=perf_counter();after=packet.observation_time_s
        origin,rotation=robot.fk(a)
        direction=np.asarray(robot.base_config[a]['base_xyz'][:2])-goals[a]
        direction=direction/max(float(np.linalg.norm(direction)),1e-9)
        candidates=[origin+np.r_[direction*back,height] for back,height in
                    ((.08,.03),(.14,.03),(.20,.06),(.10,.12))]
        def fresh_view():
            robot.hold(.1);refresh_payloads()
            if any(packet.camera_status.get(n)!='OK' for n in
                   ('scene_camera','left_wrist_camera','right_wrist_camera')):
                raise RuntimeError('PAIR_SLOT_REOBSERVE_CAMERA_UNAVAILABLE')
            view=robot.observed_workspace.slot_from_current_depth(goals[a])
            if abs(view['observation_time_s']-packet.observation_time_s)>1e-6:
                raise RuntimeError('SLOT_REOBSERVE_STALE_OR_WRONG_VIEW')
            save_observation(env.output,'slot_reobserve_'+a+'_'+str(len(entry['checks'])),packet,scene)
            return view
        def adjust(position):
            label='slot_reobserve_'+a+'_'+str(len(entry['checks']))
            check=dict(kind='viewpoint',tcp_m=position.tolist(),executed=False)
            entry['checks'].append(check)
            try:
                move('transfer',{a:(position,rotation)},active=(a,),carrying=carried_arms,
                     dry_run=True,label='preview_'+label)
            except RuntimeError as error:
                if not path_rejection(str(error)) and not str(error).startswith('UNOBSERVED_PATH_SPACE'):raise
                check['planning_rejection']=str(error);return False
            # Never catch a drive/holding failure and try another viewpoint.
            move('transfer',{a:(position,rotation)},active=(a,),carrying=carried_arms,label=label)
            check['executed']=True
            return True
        try:
            view=reobserve_reserved_slot(slots,a,goals[a],after,fresh_view,candidates,adjust,entry['checks'])
            before_transfer[goals[a]]=dict(view)
            entry.update(success=True,fresh_empty_view=dict(view))
        except Exception as error:
            entry['failure_reason']=str(error);raise
        finally:entry['wall_s']=perf_counter()-start

    def clear_empty_arm(a):
        report['last_phase']='clear_for_verification'
        if not carried_arms:
            robot.execute_joint(a,robot.home,scene,None)
            return
        # Keep the waiting loaded arm on the SAME clock and enforce both
        # its payload clearance and measured arm separation during withdrawal.
        robot.held_visual_geometry=None
        path,_=robot.checked_joint_plan(a,robot.read_states()[a].joint_positions_rad,robot.home[:7],scene,None)
        paths=[path if arm==a else [robot.read_states()[arm].joint_positions_rad]*2 for arm in arms]
        paths,_=transfer_schedule(robot,paths,carried_clearance)
        def guard(left,right):
            reason,_=check_spheres(robot.spheres(arms[0],left),robot.spheres(arms[1],right),[],None,margin=.008)
            if reason:raise RuntimeError('PAIR_SEQUENTIAL_'+reason)
            carried_clearance(left,right)
        phases['sequential_'+a+'_clear']=drive_empty_pregrasps(robot,paths,guard,
            require_overlap=False,speed_fraction=.4,loaded_transfer=True)

    def confirm_first_placement(a):
        nonlocal packet
        stable=None;previous=None
        for _ in range(20):
            robot.hold(.1)
            refresh_payloads()
            matches=[o for o in fresh_scene.objects if o.posture=='UPRIGHT' and not o.failures
                and pose_is_observed(o,fresh_scene)
                and np.linalg.norm(np.asarray(o.position_m[:2])-goals[a])<=.018
                and abs(o.position_m[2]-robot.place_support_z-o.size_m[2]/2)<=.008
                and abs((o.axis_yaw_rad+math.pi/2)%math.pi-math.pi/2)<=math.radians(8)
                and o.velocity_m_s is not None and np.linalg.norm(o.velocity_m_s)<=.015
                and np.allclose(o.size_m,objects[a].size_m,atol=.002,rtol=0)]
            if len(matches)!=1:
                stable=None;continue
            o,=matches
            if stable is None or o.track_id!=previous:stable=packet.observation_time_s
            previous=o.track_id
            if packet.observation_time_s-stable>=.5:
                report['sequential_first_placement']=dict(arm=a,track_id=o.track_id,
                    position_m=o.position_m,time_s=packet.observation_time_s,stable_duration_s=packet.observation_time_s-stable)
                return
        raise RuntimeError('PAIR_FIRST_RELEASE_NOT_STABLE:'+a)

    def finish(active):
        nonlocal packet,held
        prefix=('sequential_'+active[0]+'_') if sequential else ''
        refresh_payloads()
        update_places(active)
        verify_slots(packet,active)
        move('align',{a:(places[a][0]+[0.,0.,.18],places[a][1]) for a in active},
             active=active,carrying=carried_arms,label=prefix+'align')
        refresh_payloads()
        update_places(active)
        verify_slots(packet,active)
        move('lower',{a:places[a] for a in active},True,
             active=active,carrying=carried_arms,label=prefix+'lower')
        packet=observe()
        for a in arms:
            if a in carried_arms:measure_held(a,packet)
        for a in active:
            measurement=measurements[a]
            rotation=np.asarray(measurement['rotation'])
            height=np.asarray(measurement['position_m'])[2]-np.abs(rotation[2])@(trackers[a].size/2)
            upper_gap=height-robot.place_support_z+measurement['uncertainty_m']
            report.setdefault('release_check',{})[a]=dict(centre_m=measurement['position_m'],
                tilt_deg=math.degrees(math.acos(np.clip(rotation[2,2],-1,1))),gap_upper_m=float(upper_gap),
                measurement=measurement['status'],waypoint_reacquisition=measurement.get('waypoint_reacquisition'))
            if upper_gap>.020:raise RuntimeError('PAIR_RELEASE_HEIGHT_UNSUPPORTED:'+a)
        report['last_phase']='release'
        fingers(False,active)
        carried_arms.difference_update(active)
        held=bool(carried_arms)
        robot.held_visual_geometry=None
        for a in active:released_references[a]=measurements[a].copy()
        # A new frame after opening must reconcile the STILL HELD carton too;
        # otherwise its clipped detections become false stationary obstacles
        # for the withdrawing empty hand.
        if carried_arms:refresh_payloads()
        else:packet=observe()
        move('retreat',{a:(places[a][0]+[0.,0.,.18],places[a][1]) for a in active},True,
             active=active,carrying=carried_arms,label=prefix+'retreat')
        for a in active:clear_empty_arm(a)

    orient={}
    for a in arms:
        t=trackers[a];pos,rot=robot.fk(a)
        centre=pos+rot@t.offset;goal_rot=places[a][1]
        orient[a]=(centre-goal_rot@t.offset,goal_rot)

    def plan_orient():
        q=move('transfer_orient',orient,dry_run=True,label='preview_orient')
        dest={}
        for a in arms:
            t=trackers[a];rot=orient[a][1]
            centre=np.array([*goals[a],robot.place_support_z+t.size[2]/2+.012+.18])
            dest[a]=(centre-rot@t.offset,rot)
        move('transfer_after_orient',dest,dry_run=True,start_states=q,label='preview_after_orient')
        return orient

    def plan_sequential():
        nonlocal sequential
        sequential=True
        attempts=report.setdefault('sequential_staging_checks',[])
        initial={a:robot.fk(a) for a in arms}
        # Progress is geometric and independent of seed, box identity or truth.
        for fraction in (.8,.65,.5):
            stage={a:(initial[a][0]+fraction*(overhead[a][0]-initial[a][0]),initial[a][1]) for a in arms}
            try:q=move('transfer',stage,dry_run=True,label='preview_sequential_stage')
            except RuntimeError as error:
                if not path_rejection(str(error)):raise
                attempts.append(dict(fraction=fraction,reason=str(error)));continue
            for first in arms:
                try:
                    q1=move('transfer',{first:overhead[first]},active=(first,),dry_run=True,
                            start_states=q,label='preview_sequential_first_transfer')
                    q2=move('align',{first:(places[first][0]+[0,0,.18],places[first][1])},
                            active=(first,),dry_run=True,start_states=q1,label='preview_sequential_first_align')
                    move('lower',{first:places[first]},True,active=(first,),dry_run=True,
                            start_states=q2,label='preview_sequential_first_lower')
                except RuntimeError as error:
                    if not path_rejection(str(error)):raise
                    attempts.append(dict(fraction=fraction,first_arm=first,reason=str(error)));continue
                return dict(fraction=fraction,stage=stage,order=(first,next(a for a in arms if a!=first)))
        raise RuntimeError('NO_CLEAR_SEQUENTIAL_END_PLACEMENT')

    policy=env.c.get('pair_transfer_policy','auto')
    attempts=report.setdefault('transfer_strategy_checks',[])
    strategy,plan=choose_transfer_strategy(policy,{
        'parallel':lambda:move('transfer',overhead,dry_run=True,label='preview_parallel'),
        'orient-first':plan_orient,'sequential-end':plan_sequential},attempts)
    sequential=strategy=='sequential-end'
    report['transfer_strategy']=dict(requested=policy,selected=strategy,
        selection='first feasible deterministic plan; no learned score or duration optimization',
        execution_failure_policy='stop; never switch strategy after a drive/perception failure')
    if strategy=='parallel':
        move('transfer',overhead)
        finish(arms)
    elif strategy=='orient-first':
        report['transfer_replan']=dict(strategy='orient_at_separated_pickup_then_translate')
        move('transfer_orient',orient)
        refresh_payloads()
        dest={}
        for a in arms:
            t=trackers[a];rot=robot.fk(a)[1]
            centre=np.array([*goals[a],robot.place_support_z+t.size[2]/2+.012+.18])
            dest[a]=(centre-rot@t.offset,rot)
        move('transfer_after_orient',dest)
        finish(arms)
    else:
        report['sequential_plan']=dict(staging_fraction=plan['fraction'],order=plan['order'],
            stage_tcp_m={a:p.tolist() for a,(p,r) in plan['stage'].items()})
        move('transfer',plan['stage'],label='sequential_shared_transfer')
        if not phases['sequential_shared_transfer']['overlap_ticks']:
            raise RuntimeError('NO_MEASURED_PARALLEL_TRANSFER')
        for index,a in enumerate(plan['order']):
            refresh_payloads()
            if index==1:reobserve_second_slot(a)
            t=trackers[a];rot=robot.fk(a)[1]
            centre=np.array([*goals[a],robot.place_support_z+t.size[2]/2+.012+.18])
            move('transfer',{a:(centre-rot@t.offset,rot)},active=(a,),carrying=carried_arms,
                 label='sequential_'+a+'_transfer')
            finish((a,))
            if index==0:confirm_first_placement(a)
    # Both hands are clear. Final acceptance uses fresh unrestricted perception,
    # never the release envelopes or the first-placement intermediate check.
    released_references.clear()
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
        current_sources=match_stationary_sources(scene,source_references)
        ready=len(set(ids))==2
        if not ready:
            stable_since=None
        elif stable_since is None or ids!=previous_ids:
            stable_since=scene.observation_time_s
        previous_ids=ids
        if ready and scene.observation_time_s-stable_since>=.5:
            for a,p in placements.items():
                slots.confirm_placement(a,goals[a],p['observed_track'])
            for obj in objects.values():estimator.mark_identity(obj.identity_id,'placed')
            report['placements']=placements
            report['unselected_sources_after_placement']={k:dict(track_id=o.track_id,position_m=o.position_m)
                                                         for k,o in current_sources.items()}
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
