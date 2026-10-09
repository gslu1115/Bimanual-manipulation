"""Official GraspGen ZMQ protocol, evaluation only until TCP calibration is verified.

GraspGen returns base poses in the input cloud frame (+Z approach, +X closing).
The server centers AND restores translations; do not add object center twice.
It consumes object cloud only. Scene cloud is kept for downstream collision
checks and is not silently sent as an unsupported server field.
"""
from dataclasses import dataclass
import numpy as np
from workstation.perception.rgbd_cartons import deproject


@dataclass(frozen=True)
class VisualClouds:
    target_m: np.ndarray
    scene_m: np.ndarray
    time_s: float
    frame: str = 'workcell'
    unseen_space_is_free: bool = False


def visual_clouds(packet,scene,target_id,max_points=20000):
    target=next(o for o in scene.objects if o.track_id == target_id)
    object_points=[]; scene_points=[]
    for name,frame in packet.cameras.items():
        if frame is None or scene.camera_status.get(name) != 'OK': continue
        if abs(frame.sample_time_s-scene.observation_time_s) > .02:
            raise ValueError('Cloud/frame time mismatch')
        points=deproject(frame)
        valid=frame.valid_depth & np.isfinite(points).all(axis=-1)
        scene_points.append(points[valid])
        for ref in target.mask_refs:
            if ref.startswith(name+'/'):
                object_points.append(points[valid & scene.masks[ref]])
    if not object_points: raise ValueError('No observed target points')
    def bounded(items):
        points=np.concatenate(items).astype(np.float32)
        if len(points) > max_points:
            points=points[np.linspace(0,len(points)-1,max_points,dtype=int)]
        return points
    return VisualClouds(bounded(object_points),bounded(scene_points),scene.observation_time_s)


def validate_grasps(grasps,scores):
    poses=np.asarray(grasps,dtype=np.float64); scores=np.asarray(scores,dtype=float).reshape(-1)
    if poses.shape != (len(scores),4,4) or not np.isfinite(poses).all() or not np.isfinite(scores).all():
        raise ValueError('Malformed GraspGen output')
    if len(poses) and (not np.allclose(poses[:,3,:],[0,0,0,1],atol=1e-5) or
        not np.allclose(poses[:,:3,:3]@poses[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-3) or
        not np.allclose(np.linalg.det(poses[:,:3,:3]),1.,atol=1e-3)):
        raise ValueError('Invalid rigid grasp pose')
    return poses,scores


def tcp_poses(base_poses,T_graspgen_base_from_project_tcp,calibration_verified=False):
    if not calibration_verified:
        raise ValueError('GraspGen/project TCP calibration has not been verified')
    transform,_=validate_grasps(np.asarray(T_graspgen_base_from_project_tcp)[None],[0.])
    return np.asarray(base_poses)@transform[0]


class GraspGenClient:
    def __init__(self,endpoint='tcp://127.0.0.1:5557',timeout_ms=3000):
        self.endpoint,self.timeout_ms=endpoint,timeout_ms

    def request(self,payload):
        # Optional dependencies live in the isolated inference environment.
        import zmq
        import msgpack
        import msgpack_numpy
        context=zmq.Context()
        socket=context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER,0)
        socket.setsockopt(zmq.RCVTIMEO,self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO,self.timeout_ms)
        try:
            socket.connect(self.endpoint)
            socket.send(msgpack.packb(payload,default=msgpack_numpy.encode,use_bin_type=True))
            response=msgpack.unpackb(socket.recv(),object_hook=msgpack_numpy.decode,raw=False)
            if 'error' in response: raise RuntimeError(response['error'])
            return response
        except zmq.Again:
            raise RuntimeError('GRASPGEN_SERVICE_UNAVAILABLE:'+self.endpoint) from None
        finally:
            socket.close(); context.term()

    def metadata(self):
        metadata=self.request({'action':'metadata'})
        if metadata.get('gripper_name') != 'franka_panda':
            raise ValueError('GraspGen service is not configured for Franka Panda')
        return metadata

    def infer(self,clouds,num_grasps=64):
        if clouds.frame != 'workcell' or clouds.unseen_space_is_free:
            raise ValueError('Expected observed metre workcell clouds with unknown unseen space')
        points=np.asarray(clouds.target_m,np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 200 or not np.isfinite(points).all():
            raise ValueError('Insufficient or malformed observed object cloud')
        if np.max(np.ptp(points,axis=0)) > 1.:
            raise ValueError('Carton cloud extent suggests a unit/frame error')
        self.metadata()
        response=self.request(dict(action='infer',point_cloud=points,num_grasps=num_grasps,
            topk_num_grasps=20,min_grasps=1,max_tries=1,grasp_threshold=-1.,remove_outliers=True))
        poses,scores=validate_grasps(response['grasps'],response['confidences'])
        return dict(base_poses_workcell=poses,scores=scores,scores_are_success_probability=False,
                    robot_path_checked=False,TCP_calibration_verified=False,timing=response.get('timing',{}))
