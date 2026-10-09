"""Camera inspection goals through image-time calibrated hand/tool transforms."""
import numpy as np


def rigid_transform(position,rotation):
    result=np.eye(4);result[:3,:3]=rotation;result[:3,3]=position
    return result


def camera_goal_to_tcp(T_world_hand_image,T_world_tcp_image,T_world_camera_image,T_world_camera_goal):
    transforms=[np.asarray(t,float) for t in
                (T_world_hand_image,T_world_tcp_image,T_world_camera_image,T_world_camera_goal)]
    for t in transforms:
        if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
            raise ValueError('Invalid calibrated rigid transform')
        if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5):
            raise ValueError('Non-rigid camera transform')
    hand,tcp,camera,goal=transforms
    T_hand_camera=np.linalg.inv(hand)@camera
    T_hand_tcp=np.linalg.inv(hand)@tcp
    return goal@np.linalg.inv(T_hand_camera)@T_hand_tcp,T_hand_camera
