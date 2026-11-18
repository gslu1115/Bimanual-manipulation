"""Inputs for one MoveIt joint-group request, from the current depth map and encoders."""
from copy import deepcopy
from dataclasses import asdict
import math
import json
import socket
import time
import xml.etree.ElementTree as ET
import numpy as np

ARMS = ('panda_left', 'panda_right')


def description(urdf, robot, states):
    source = ET.parse(urdf).getroot()
    parents = {j.find('child').get('link'): j for j in source.findall('joint')}
    joint_names = []
    description = ET.Element('robot', name='dual_panda')
    semantic = ET.Element('robot', name='dual_panda')
    group = ET.SubElement(semantic, 'group', name='both_arms')
    ET.SubElement(description, 'link', name='workcell')
    for arm in ARMS:
        side = arm.split('_')[1] + '_'
        base = robot.base_config[arm]
        selected = {'base_link', 'right_gripper', *(name for name, *_ in robot.sphere_description)}
        for frame in tuple(selected):
            while frame != 'base_link':
                frame = parents[frame].find('parent').get('link')
                selected.add(frame)
        links = {name: ET.SubElement(description, 'link', name=side+name) for name in sorted(selected)}
        for name in sorted(selected - {'base_link'}):
            joint = deepcopy(parents[name])
            if joint.get('type')=='revolute' and hasattr(robot,'joint_velocity_limits'):
                idx=int(joint.get('name').removeprefix('panda_joint'))-1
                joint.find('limit').set('velocity',str(robot.joint_velocity_limits[arm][idx]))
            joint.set('name', side+joint.get('name'))
            for tag in ('parent','child'):
                node=joint.find(tag)
                node.set('link',side+node.get('link'))
            description.append(joint)
        mount=ET.SubElement(description,'joint',name=side+'mount',type='fixed')
        ET.SubElement(mount,'parent',link='workcell')
        ET.SubElement(mount,'child',link=side+'base_link')
        ET.SubElement(mount,'origin',xyz=' '.join(map(str,base['base_xyz'])),rpy=f"0 0 {math.radians(base['yaw_deg'])}")
        finger='finger_envelope'
        links[finger]=ET.SubElement(description,'link',name=side+finger)
        fixed=ET.SubElement(description,'joint',name=side+finger,type='fixed')
        ET.SubElement(fixed,'parent',link=side+'panda_hand')
        ET.SubElement(fixed,'child',link=side+finger)
        width=float(np.max(states[arm].finger_positions_m))
        spheres=list(robot.sphere_description)+[(finger,9,(0,sign*(width+.004),z),.009)
                for sign in (-1,1) for z in (.060,.080,.095)]
        indices={}
        for name,index,centre,radius in spheres:
            indices[name]=index
            body=ET.SubElement(links[name],'collision')
            ET.SubElement(body,'origin',xyz=' '.join(map(str,centre)))
            ET.SubElement(ET.SubElement(body,'geometry'),'sphere',radius=str(radius))
        for left,li in indices.items():
            for right,ri in indices.items():
                if left < right and abs(li-ri)<=2:
                    ET.SubElement(semantic,'disable_collisions',link1=side+left,link2=side+right,reason='Adjacent')
        for index in range(1,8):
            name=side+f'panda_joint{index}'
            ET.SubElement(group,'joint',name=name)
            joint_names.append(name)
    return ET.tostring(description,encoding='unicode'),ET.tostring(semantic,encoding='unicode'),joint_names


def request(urdf,robot,states,goals,acceleration_limits):
    workspace=robot.observed_workspace
    workspace._build_occupied()
    key=workspace._mapping_key()
    chunks=workspace.frame_cache['fixed_points'][key[:9]+key[10:12]]
    # The OctoMap also gets the visually modelled carton surfaces. Only the
    # fixed station and the robot at image time have been filtered out.
    cells=np.unique(np.floor(np.concatenate(chunks)/workspace.resolution).astype(int),axis=0)
    centres=(cells+.5)*workspace.resolution
    urdf,srdf,names=description(urdf,robot,states)
    return dict(urdf=urdf,srdf=srdf,joint_names=names,
                acceleration_limits=list(acceleration_limits),
                start=np.concatenate([states[arm].joint_positions_rad for arm in ARMS]).tolist(),
                goal=np.concatenate(goals).tolist(),resolution=workspace.resolution,
                occupied=centres.tolist(),boxes=[asdict(b) for b in (*workspace.static,*workspace.objects)],
                midline_x=float(np.mean([robot.base_config[a]['base_xyz'][0] for a in ARMS])),
                frame_ids={n:f.sequence_id for n,f in workspace.packet.cameras.items() if f is not None})


def checked_approach(robot,scene,pair,states,goals):
    from workstation.diagnostics.graspgen_pair_place import path_rejection
    start=time.perf_counter()
    payload=request(robot.moveit_urdf,robot,states,goals,robot.moveit_acceleration)
    checked_s=0.
    rejected_direct=None
    for prefer_direct in (True,False):
        payload['prefer_direct']=prefer_direct
        result,paths,ts=_timed_approach(robot.moveit_socket,payload)
        checked_at=time.perf_counter()
        try:
            minimum,count=_check_waypoints(robot,scene,pair,paths)
        except RuntimeError as error:
            if not prefer_direct or result.get('route')!='direct' or not path_rejection(str(error)):raise
            rejected_direct=str(error)
            continue
        finally:
            checked_s+=time.perf_counter()-checked_at
        return paths,ts.tolist(),minimum,count,dict(search_s=result['search_s'],sampler_calls=result['sampler_calls'],
            waypoints=len(ts),depth_voxels=len(payload['occupied']),frame_ids=payload['frame_ids'],
            route=result.get('route','rrt_connect'),direct_check_s=result.get('direct_check_s'),
            direct_project_rejection=rejected_direct,request_and_map_wall_s=time.perf_counter()-start-checked_s,
            project_check_wall_s=checked_s)


def _timed_approach(address,payload):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.settimeout(45)
        client.connect(address)
        client.sendall(json.dumps(payload,allow_nan=False).encode()+b'\n')
        with client.makefile('rb') as stream:
            result=json.loads(stream.readline())
    if not result['ok']:raise RuntimeError('MOVEIT_PAIR: '+result['error'])
    q=np.asarray(result['positions'],dtype=float)
    ts=np.asarray(result['times_s'],dtype=float)
    if (q.ndim!=2 or q.shape!=(len(ts),14) or len(q)<2 or not np.isfinite(q).all()
        or not np.isfinite(ts).all() or ts[0]!=0 or np.any(np.diff(ts)<=0)
        or np.max(np.abs(q[0]-payload['start']))>1e-5 or np.max(np.abs(q[-1]-payload['goal']))>.001):
        raise RuntimeError('MOVEIT_PAIR_INVALID_TRAJECTORY')
    return result,q.reshape(-1,2,7).transpose(1,0,2),ts


def _check_waypoints(robot,scene,pair,paths):
    from workstation.diagnostics.graspgen_pair import synchronized_knots
    minimum=float('inf')
    for count,(left,right) in enumerate(synchronized_knots(*paths),1):
        for a,row,own,other in zip(ARMS,pair,(left,right),(right,left)):
            gap=robot.validate_path(a,[own,own],scene,row.candidate.target_id,other_joint_positions_rad=other,
                                    validation_stage='moveit_joint_approach')
            minimum=min(minimum,gap)
    return minimum,count


def timed_knots(paths,times,dt,max_speed):
    q=np.stack(paths,axis=1)
    durations=np.maximum(np.diff(times),np.max(np.abs(np.diff(q,axis=0)),axis=(1,2))/max_speed)
    clock=np.r_[0.,np.cumsum(durations)]
    ticks=math.ceil(clock[-1]/dt)
    grid=np.linspace(0.,clock[-1],ticks+1)
    return np.stack([np.interp(grid,clock,col) for col in q.reshape(len(q),-1).T],axis=1).reshape(-1,2,7)
