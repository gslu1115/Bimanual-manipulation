"""Verify static gripper geometry and Lula frames before composing model TCPs.

This is a simulation asset calibration, not a physical hand-eye calibration.
It never moves either robot and does not make a candidate executable.
"""
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def stl_vertices(path):
    data=Path(path).read_bytes();count=int.from_bytes(data[80:84],'little')
    if len(data)!=84+50*count:raise ValueError('Expected binary STL asset')
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    return np.unique(np.frombuffer(data,offset=84,count=count,dtype=dtype)['vertices'].reshape(-1,3),axis=0).astype(float)

def align_gripper_meshes(model_vertices,robot_vertices,tolerance_m=1e-5):
    """Derive the base frame from the rigid palm; audit finger geometry separately."""
    checks=[]
    for degrees in (0,90,-90,180):
        angle=np.deg2rad(degrees);c,s=np.cos(angle),np.sin(angle)
        r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        errors={}
        for name in ('hand','finger'):
            a=np.asarray(model_vertices[name]);b=np.asarray(robot_vertices[name])@r.T
            errors[name]=float(max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max()))
        checks.append(dict(degrees=degrees,maximum_vertex_distance_m=errors))
    matches=[x for x in checks if x['maximum_vertex_distance_m']['hand']<=tolerance_m]
    if len(matches)!=1:raise ValueError('Gripper mesh alignment is missing or ambiguous: '+str(checks))
    angle=np.deg2rad(matches[0]['degrees']);c,s=np.cos(angle),np.sin(angle)
    transform=np.eye(4);transform[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    return transform,checks

def relative_tcp_from_fk(fk,config):
    """Measure the same hand-to-TCP transform on both arms and several FK poses."""
    measurements=[]
    for item in config['robots']:
        arm=item['name'];home=np.asarray(config['robot_home'][:7],float)
        for delta in (0.,.1,-.1):
            q=home.copy();q[0]+=delta
            hp,hr=fk(arm,q,'panda_hand');tp,tr=fk(arm,q,'right_gripper')
            hand=np.eye(4);hand[:3,:3]=hr;hand[:3,3]=hp
            tcp=np.eye(4);tcp[:3,:3]=tr;tcp[:3,3]=tp
            measurements.append(np.linalg.inv(hand)@tcp)
    reference=measurements[0]
    max_error=max(float(np.max(np.abs(x-reference))) for x in measurements)
    if max_error>1e-6:raise ValueError('Hand/TCP transform is not fixed across arms and poses')
    return reference,dict(samples=len(measurements),maximum_matrix_error=max_error)

def verify(env,robot,isaac_root,graspgen_root):
    import yaml
    config_path=Path(graspgen_root)/'config/grippers/franka_panda.yaml'
    gripper_config=yaml.safe_load(config_path.read_text())
    offset=gripper_config.get('transform_offset_from_asset_to_graspgen_convention')
    if offset != [[0.,0.,0.],[0.,0.,0.,1.]]:
        raise ValueError('This measured calibration requires the verified identity asset offset')
    model_dir=Path(graspgen_root)/'assets/panda_gripper'
    robot_dir=Path(isaac_root)/'exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/meshes/collision'
    paths={side:{name:folder/(name+'.stl') for name in ('hand','finger')}
           for side,folder in (('model',model_dir),('robot',robot_dir))}
    transform,checks=align_gripper_meshes(
        {name:stl_vertices(path) for name,path in paths['model'].items()},
        {name:stl_vertices(path) for name,path in paths['robot'].items()})
    hand_tcp,fk_check=relative_tcp_from_fk(robot.fk,env.c)
    finger_error=min(checks,key=lambda x:x['maximum_vertex_distance_m']['hand'])['maximum_vertex_distance_m']['finger']
    result=dict(scope='rigid_palm_frame_and_Lula_FK_only',verified=True,
        gripper_geometry_equivalence_verified=finger_error<=1e-5,
        finger_maximum_nearest_vertex_distance_m=finger_error,
        candidate_execution_authorized=False,
        T_graspgen_base_from_panda_hand=transform.tolist(),T_panda_hand_from_project_tcp=hand_tcp.tolist(),
        T_graspgen_base_from_project_tcp=(transform@hand_tcp).tolist(),mesh_checks=checks,fk_check=fk_check,
        gripper_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        asset_sha256={side:{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in group.items()}
                      for side,group in paths.items()},physical_execution=False,real_robot_calibration=False)
    env._write_json('graspgen_tcp_calibration.json',result)
    return result
