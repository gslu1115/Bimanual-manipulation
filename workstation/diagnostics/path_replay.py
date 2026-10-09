"""Replay saved camera/encoder/visual estimates; no physics or carton truth.

Run with Isaac Python. SimulationApp is needed to load the existing Lula API;
the replay does not construct a world or drive any robot.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--prefix',default='pregrasp')
    parser.add_argument('--diagnose-rejection',action='store_true')
    parser.add_argument('--audit-held-history',action='store_true',
                        help='saved masks/optical rays and payload self shadow; no robot commands')
    parser.add_argument('--associate-supported-track',action='store_true',
                        help='reconcile current whole-instance holding support with the original visual ID')
    parser.add_argument('--check-held-lift',action='store_true',
                        help='with held history: validate full measured-attitude lift, no drive')
    parser.add_argument('--check-held-lower',action='store_true',
                        help='with held history: validate full slot descent, no drive')
    args=parser.parse_args()
    if (args.check_held_lift or args.check_held_lower) and not args.audit_held_history:
        parser.error('held motion checks require --audit-held-history')
    if args.check_held_lift and args.check_held_lower:
        parser.error('select one held motion check')
    if args.associate_supported_track and not args.audit_held_history:
        parser.error('supported identity association requires current holding audit')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True})
    try:
        import numpy as np
        from workstation.observations.observation_packet import CameraFrame,RobotState,FORMAL_CAMERA_NAMES
        from workstation.perception.scene_estimate import ScenePriors
        from workstation.simulation.robot_driver import RobotMotion
        folder=args.run_dir;frames={}
        for name in FORMAL_CAMERA_NAMES:
            with np.load(folder/(args.prefix+'_'+name+'.npz')) as data:
                states={str(name):RobotState(float(time),q[:7],q[7:],v[:7],v[7:]) for name,time,q,v in
                        zip(data['robot_names'],data['robot_time_s'],data['robot_qpos'],data['robot_qvel'])}
                frames[name]=CameraFrame(name,data['rgb'],data['depth_m'],data['valid_depth'],data['K'],
                    data['T_workcell_from_camera_cv'],float(data['time_s']),int(data['sequence_id']),states)
        config=json.loads((folder/'config_used.json').read_text(encoding='utf-8'))
        raw=json.loads((folder/(args.prefix+'_scene.json')).read_text(encoding='utf-8'))
        masks={}
        mask_file=folder/(args.prefix+'_masks.npz')
        if mask_file.is_file():
            with np.load(mask_file,allow_pickle=False) as data:masks={ref:data[ref] for ref in data.files}
        from workstation.perception.scene_estimate import SceneEstimate,ObjectEstimate,PoseHypothesis,ObservedEdge
        objects=[]
        for item in raw['objects']:
            item=dict(item);item['hypotheses']=tuple(PoseHypothesis(**hyp) for hyp in item['hypotheses'])
            for field in ('position_m','size_m','mask_refs','cameras','failures'):item[field]=tuple(item[field])
            objects.append(ObjectEstimate(**item))
        scene=SceneEstimate(raw['observation_time_s'],raw['assembled_time_s'],raw['frame'],tuple(objects),
            tuple(raw['failures']),raw['camera_status'],masks,tuple(ObservedEdge(**edge) for edge in raw.get('edges',())))
        time=raw['observation_time_s'];states=frames['scene_camera'].robot_state_at_frame
        packet=SimpleNamespace(cameras=frames,camera_status={name:'OK' for name in frames},
            observation_time_s=time,assembled_time_s=time,robot_state=states)
        def no_drive(*args):raise RuntimeError('Offline replay cannot drive robots')
        robot=RobotMotion(config,lambda:states,no_drive,no_drive)
        robot.set_observation(packet,scene,ScenePriors.from_config(config))
        report=json.loads((folder/'visual_task_report.json').read_text(encoding='utf-8'))
        decision=report['decision'].get('candidate',report['assessments'][0]['candidate'])
        arm=decision['arm'];target=next((obj for obj in scene.objects if obj.track_id==decision['target_id']),None)
        if target is None:
            if not args.associate_supported_track:raise ValueError('Current visual target ID absent; explicit supported-track audit required')
            closed=json.loads((folder/'closed_scene.json').read_text(encoding='utf-8'))
            # Original visual latch only. Never return it as a current scene pose.
            target=next(SimpleNamespace(**obj) for obj in closed['objects'] if obj['track_id']==decision['target_id'])
        p,_=robot.fk(arm);yaw=decision['yaw_rad']
        from workstation.planning.selection import upright_grasp_position
        goal=upright_grasp_position(target.position_m)
        if args.audit_held_history:
            from workstation.perception.held_surface import HeldSurfaceTracker
            from workstation.planning.observed_workspace import ObservedWorkspace
            if not masks:raise ValueError('Exact saved masks required; do not infer them from overlay PNG')
            tcp,r=robot.fk(arm,states[arm].joint_positions_rad)
            # Reuse the ORIGINAL visual latch, not a new pose fitted to a later
            # clipped view. In-flight SceneEstimate may retain ambiguous pose.
            closed=json.loads((folder/'closed_scene.json').read_text(encoding='utf-8'))
            latch_obj=next(SimpleNamespace(**o) for o in closed['objects'] if o['track_id']==target.track_id)
            with np.load(folder/('closed_scene_camera.npz')) as data:
                latch_q=next(q[:7] for name,q in zip(data['robot_names'],data['robot_qpos']) if str(name)==arm)
            latch_tcp,latch_r=robot.fk(arm,latch_q)
            tracker=HeldSurfaceTracker(latch_obj,latch_tcp,latch_r,ScenePriors.from_config(config).carton_size_m)
            robot.held_visual_geometry=dict(arm=arm,target_id=target.track_id,
                                           offset=tracker.offset,relative_rotation=tracker.relative_rotation,
                                           uncertainty_m=tracker.uncertainty_m)
            measurement=tracker.observe(packet,scene,tcp,r)
            if measurement is None:raise ValueError('Saved closed frame does not support current payload hypothesis')
            old_ids=[obj.track_id for obj in scene.objects]
            if args.associate_supported_track:
                from workstation.perception.rgbd_cartons import CartonEstimator
                scene=CartonEstimator(ScenePriors.from_config(config)).associate_supported_target(scene,target.track_id,measurement)
                if not any(obj.track_id==target.track_id for obj in scene.objects):
                    raise ValueError('Current whole-instance support cannot associate the original visual target')
                robot.set_observation(packet,scene,ScenePriors.from_config(config))
            confirmed=robot.confirm_payload_observation(target.track_id,tracker,measurement,packet)
            if not confirmed:raise ValueError('Current payload observation rejected by workspace')
            detail=report.get('failure_path_check',{})
            point=detail.get('workspace',{}).get('last_unknown_sample_m')
            if point is None:
                q=detail.get('candidate_joint_positions_rad',states[arm].joint_positions_rad)
                point=robot.fk(arm,q)[0];query_source='encoder FK of recorded rejection/current state; not an unknown-cell diagnosis'
            else:query_source='recorded unknown-space rejection sample'
            point=np.asarray(point)
            points=np.tile(point,(9,1));points[:,2]+=np.linspace(0,.18,9)
            anchors=[]
            for prefix in ('initial','tracked'):
                previous={}
                for name in FORMAL_CAMERA_NAMES:
                    with np.load(folder/(prefix+'_'+name+'.npz')) as data:
                        old_states={str(n):RobotState(float(t),q[:7],q[7:],v[:7],v[7:]) for n,t,q,v in
                            zip(data['robot_names'],data['robot_time_s'],data['robot_qpos'],data['robot_qvel'])}
                        previous[name]=CameraFrame(name,data['rgb'],data['depth_m'],data['valid_depth'],data['K'],
                            data['T_workcell_from_camera_cv'],float(data['time_s']),int(data['sequence_id']),old_states)
                old_raw=json.loads((folder/(prefix+'_scene.json')).read_text(encoding='utf-8'))
                old_time=old_raw['observation_time_s']
                old_packet=SimpleNamespace(cameras=previous,camera_status=old_raw['camera_status'],
                    observation_time_s=old_time,assembled_time_s=old_time)
                old_scene=SimpleNamespace(objects=tuple(SimpleNamespace(**obj) for obj in old_raw['objects']))
                old=ObservedWorkspace(old_packet,robot.observed_workspace.priors,old_scene)
                robot.occlusion_memory.anchor(old)
                anchors.append(dict(prefix=prefix,age_s=time-old_time,
                    measured_ray_free={n:old._ray_free(points,f).tolist() for n,f in previous.items()},
                    outside_other_motion_envelopes=robot.occlusion_memory._outside_motion_envelopes(
                        points,old,robot.observed_workspace,time-old_time,target.track_id).tolist()))
            views=[]
            for name,frame in frames.items():
                t=frame.T_workcell_from_camera_cv;camera=(point-t[:3,3])@t[:3,:3];uv=frame.K@camera;uv=uv[:2]/uv[2]
                u,v=np.rint(uv).astype(int);h,w=frame.depth_m.shape
                entry=dict(camera=name,uv=uv.tolist())
                if camera[2]>0 and 1<=u<w-1 and 1<=v<h-1:
                    memory=robot.occlusion_memory;ws=robot.observed_workspace
                    entry.update(robot_footprint=bool(memory._robot_depth_mask(ws,frame)[v,u]),
                        payload_footprint=bool(memory._payload_depth_mask(ws,frame)[v,u]))
                    from workstation.perception.rgbd_cartons import deproject
                    xyz=deproject(frame)[v-1:v+2,u-1:u+2].reshape(-1,3)
                    raw_robot=np.zeros(9,bool);nearest_plane=np.full(9,np.inf)
                    for model,pos,rot in ws.robot_links:
                        raw_robot|=model.contains(xyz,pos,rot)
                        nearest_plane=np.minimum(nearest_plane,np.max(((xyz-pos)@rot)@model.planes[:,:3].T+
                            model.planes[:,3],axis=1))
                    payload=ws.payload_observation;local=(xyz-payload['centre'])@payload['rotation']
                    raw_payload=np.all(np.abs(local)<=payload['size']/2+.006,axis=1)&(
                        np.min(np.abs(np.abs(local)-payload['size']/2),axis=1)<=.006)
                    geometry_support=raw_payload.copy()
                    segment=np.zeros(9,bool)
                    for ref in payload['mask_refs']:
                        if ref.startswith(name+'/'):segment|=scene.masks[ref][v-1:v+2,u-1:u+2].reshape(-1)
                    raw_payload&=segment
                    depth=frame.depth_m[v-1:v+2,u-1:u+2]
                    measured_free=camera[2]<depth-ws.priors.optical_depth_error_margin_m
                    self_pixels=(memory._robot_depth_mask(ws,frame,erode=False)|
                                 memory._payload_depth_mask(ws,frame,erode=False))[v-1:v+2,u-1:u+2]
                    near_fixed=(depth>=camera[2])&(depth<=camera[2]+ws.priors.optical_depth_error_margin_m)&\
                        memory._fixed_surface_depth_mask(ws,frame)[v-1:v+2,u-1:u+2]
                    entry.update(robot_pixels=raw_robot.reshape(3,3).tolist(),
                        payload_pixels=raw_payload.reshape(3,3).tolist(),
                        payload_geometry_pixels=geometry_support.reshape(3,3).tolist(),
                        payload_surface_distance_m=np.min(np.abs(np.abs(local)-payload['size']/2),axis=1).reshape(3,3).tolist(),
                        payload_box_excess_m=np.max(np.abs(local)-payload['size']/2,axis=1).reshape(3,3).tolist(),
                        instance_pixels=segment.reshape(3,3).tolist(),
                        union_footprint_all_supported=bool((raw_robot|raw_payload).all()),
                        depth_m=depth.tolist(),measured_free_pixels=measured_free.tolist(),
                        near_fixed_surface_pixels=near_fixed.tolist(),
                        mixed_footprint_supported=bool((self_pixels|measured_free|near_fixed).all() and
                            frame.valid_depth[v-1:v+2,u-1:u+2].all()),
                        nearest_mesh_plane_m=nearest_plane.reshape(3,3).tolist())
                views.append(entry)
            result=dict(scope='offline saved payload/optical history; no motion or privileged target data',
                payload_confirmed=confirmed,query_points_m=points.tolist(),anchors=anchors,
                current_self_shadow=robot.occlusion_memory._self_shadow(points,robot.observed_workspace).tolist(),
                memory_reusable=robot.occlusion_memory.free(points,robot.observed_workspace).tolist(),views=views,
                query_source=query_source,original_current_track_ids=old_ids,
                reconciled_current_track_ids=[obj.track_id for obj in scene.objects],
                identity_association=args.associate_supported_track,holding_measurement=measurement)
            if args.check_held_lift or args.check_held_lower:
                from workstation.observations.camera_geometry import quaternion_from_matrix
                if args.check_held_lift:
                    carry_goal=tcp+[0,0,.18];carry_yaw=yaw;speed=.06;key='full_lift_check'
                else:
                    carry_goal=np.array([*report['slot_xy'],robot.place_support_z+tracker.size[2]/2+robot.place_tcp_clearance_m])
                    carry_yaw=0.;speed=.05;key='full_lower_check'
                preserve=np.linalg.norm(carry_goal[:2]-tcp[:2])<1e-6
                orientations={} if not preserve else dict(orientation0=quaternion_from_matrix(r),orientation1=quaternion_from_matrix(r))
                try:
                    path=robot.cartesian_knots(arm,tcp,carry_goal,carry_yaw,carry_yaw,speed,**orientations)
                    robot.validate_path(arm,path,scene,target.track_id,tracker.size,carry_yaw,True)
                    result[key]=dict(passed=True,target_m=carry_goal.tolist(),path_check=robot.last_plan_check)
                except RuntimeError as exc:
                    result[key]=dict(passed=False,reason=str(exc),target_m=carry_goal.tolist(),path_check=robot.last_plan_check)
        elif args.diagnose_rejection:
            detail=report['assessments'][0]['terms']['rejection_details']
            detail=detail.get('joint_search',{}).get('straight_path_details',detail)
            point=np.asarray(detail['workspace']['last_unknown_sample_m'])
            result=dict(passed=False,original_rejection=detail,
                image_time_body=robot.unknown_body_diagnostic(arm,states[arm].joint_positions_rad,point),
                candidate_body=robot.unknown_body_diagnostic(arm,detail['candidate_joint_positions_rad'],point))
        else:
            knots=robot.cartesian_knots(arm,p,goal,yaw,yaw)
            try:
                robot.validate_path(arm,knots,scene,target.track_id,target_contact=True)
                result=dict(passed=True,path_check=robot.last_plan_check)
            except RuntimeError as exc:
                result=dict(passed=False,reason=str(exc),path_check=robot.last_plan_check)
        root=Path(__file__).resolve().parents[1]
        hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                ('simulation/robot_driver.py','planning/robot_geometry.py','planning/observed_workspace.py',
                 'planning/occlusion_memory.py','perception/held_surface.py','perception/rgbd_cartons.py','diagnostics/path_replay.py')}
        digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        result.update(source_sha256=hashes)
        if not args.audit_held_history:result['scope']='saved visual and encoder APPROACH replay; no physical trial'
        kind='held_lower_' if args.check_held_lower else 'held_lift_' if args.check_held_lift else 'held_history_' if args.audit_held_history else ''
        if args.associate_supported_track:kind='associated_'+kind
        output=folder/('path_replay_'+kind+args.prefix+'_'+digest[:10]+'.json')
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
    finally:app.close()


if __name__=='__main__':main()
