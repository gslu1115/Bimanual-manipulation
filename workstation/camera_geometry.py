"""Pinhole calibration. Matrices use column vectors; quaternions use wxyz."""
import numpy as np
from task_logic import rotation

USD_FROM_OPENCV = np.diag([1., -1., -1., 1.])


def quaternion_from_matrix(matrix):
    m = np.asarray(matrix)[:3, :3]
    # Symmetric eigenproblem handles 180-degree rotations without dividing by w.
    k = np.array([[m[0, 0]-m[1, 1]-m[2, 2], m[1, 0]+m[0, 1], m[2, 0]+m[0, 2], m[2, 1]-m[1, 2]],
                  [m[1, 0]+m[0, 1], m[1, 1]-m[0, 0]-m[2, 2], m[2, 1]+m[1, 2], m[0, 2]-m[2, 0]],
                  [m[2, 0]+m[0, 2], m[2, 1]+m[1, 2], m[2, 2]-m[0, 0]-m[1, 1], m[1, 0]-m[0, 1]],
                  [m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1], np.trace(m)]])/3.
    values, vectors = np.linalg.eigh(k)
    q = vectors[:, np.argmax(values)][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def pose_matrix(position, orientation_wxyz):
    result = np.eye(4)
    result[:3, :3] = rotation(orientation_wxyz)
    result[:3, 3] = position
    return result


def look_at_quaternion(position, target, up):
    forward = np.asarray(target, dtype=float)-position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return quaternion_from_matrix(np.column_stack([right, np.cross(right, forward), -forward]))


def calibration_from_params(params, resolution):
    # Replicator's view transform is WORLD->USD CAMERA in USD row-vector order.
    # Invert then transpose; USD camera looks along -Z, OpenCV along +Z.
    view = np.asarray(params['cameraViewTransform'], dtype=float).reshape(4, 4)
    world_from_usd = np.linalg.inv(view).T
    if not np.isfinite(world_from_usd).all():
        raise RuntimeError('Camera view transform is not finite')
    world_from_cv = world_from_usd @ USD_FROM_OPENCV
    w, h = resolution
    focal = float(params['cameraFocalLength'])
    aperture = np.asarray(params['cameraAperture'], dtype=float)
    if focal <= 0 or aperture.shape != (2,) or not (aperture > 0).all():
        raise RuntimeError('Invalid rendered camera lens parameters')
    offset = np.asarray(params.get('cameraApertureOffset', [0., 0.]), dtype=float)
    k = np.array([[focal/aperture[0]*w, 0., w/2-offset[0]/aperture[0]*w],
                  [0., focal/aperture[1]*h, h/2+offset[1]/aperture[1]*h], [0., 0., 1.]])
    return dict(K=k.tolist(), resolution_wh=[w, h],
                T_world_from_camera_usd=world_from_usd.tolist(),
                T_world_from_camera_opencv=world_from_cv.tolist(),
                T_camera_opencv_from_world=np.linalg.inv(world_from_cv).tolist(),
                world_translation_m=world_from_usd[:3, 3].tolist(),
                world_orientation_wxyz_usd=quaternion_from_matrix(world_from_usd).tolist(),
                camera_axes='USD: +X right, +Y up, -Z forward; OpenCV: +X right, +Y down, +Z forward',
                matrix_convention='column vectors; translation in metres',
                depth='distance_to_image_plane; metres; optical +Z; inf means background',
                distortion=[0., 0., 0., 0., 0.], pose_source='rendered_camera_params')
