import copy
import json
from pathlib import Path
import unittest

import numpy as np
from workstation.observations.camera_config import validate_cameras, camera_settings, FORMAL_CAMERAS
from workstation.observations.camera_geometry import calibration_from_params, pose_matrix, look_at_quaternion


class CameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((Path(__file__).resolve().parents[1]/'config'/'scene.json').read_text(encoding='utf-8'))

    def test_render_view_inverse_and_optical_axes(self):
        world = pose_matrix([.4, -.2, 1.3], [0., 1., 0., 0.])
        frame = calibration_from_params(dict(cameraViewTransform=np.linalg.inv(world).T.ravel(),
            cameraFocalLength=24., cameraAperture=[36., 27.]), [640, 480])
        np.testing.assert_allclose(frame['T_world_from_camera_usd'], world)
        np.testing.assert_allclose(frame['world_translation_m'], [.4, -.2, 1.3])
        cv = np.asarray(frame['T_world_from_camera_opencv'])
        np.testing.assert_allclose(cv[:3, 2], -world[:3, 2])
        np.testing.assert_allclose(np.asarray(frame['T_camera_opencv_from_world'])@cv, np.eye(4))
        np.testing.assert_allclose(frame['K'], [[426.6666666667, 0, 319.5], [0, 426.6666666667, 239.5], [0, 0, 1]])

    def test_wrist_quaternions_look_towards_fingers_with_consistent_image_right(self):
        cameras = {x['name']: x for x in self.c['cameras']}
        for name in ('left_wrist_camera', 'right_wrist_camera'):
            c = cameras[name]
            matrix = pose_matrix(c['mount_translation'], c['mount_orientation_wxyz'])
            expected = np.array([0., 0., .125])-c['mount_translation']
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(-matrix[:3, 2], expected, atol=1e-9)
            np.testing.assert_allclose(matrix[:3, 0], [0., 1., 0.], atol=1e-9)

    def test_scene_is_oblique_and_aimed_at_shared_work_region(self):
        c = next(x for x in self.c['cameras'] if x['name'] == 'scene_camera')
        q = look_at_quaternion(c['position'], c['look_at'], c['up'])
        matrix = pose_matrix(c['position'], q)
        ray = -matrix[:3, 2]
        expected = np.asarray(c['look_at'])-c['position']
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(ray, expected, atol=1e-9)
        self.assertLess(ray[2], 0.)
        self.assertGreater(np.linalg.norm(ray[:2]), 0.)

    def test_three_camera_contract_and_performance_switches(self):
        self.c['camera_system']['save_images'] = False
        validate_cameras(self.c)
        self.assertEqual(camera_settings(self.c)['render_interval_steps'], 4)
        self.assertEqual({x['name'] for x in self.c['cameras']}, set(FORMAL_CAMERAS))
        extra = copy.deepcopy(self.c)
        camera = copy.deepcopy(extra['cameras'][0])
        camera['name'] = 'overhead'
        extra['cameras'].append(camera)
        with self.assertRaises(ValueError): validate_cameras(extra)
        missing = copy.deepcopy(self.c)
        missing['cameras'].pop()
        with self.assertRaises(ValueError): validate_cameras(missing)

    def test_reject_wrong_mount_and_ambiguous_camera_config(self):
        for key, value in [('robot', 'unknown'), ('mount_orientation_wxyz', [1, 1, 0, 0]),
                           ('clipping_range_m', [2, .01]), ('annotations', ['unsupported'])]:
            c = copy.deepcopy(self.c)
            next(x for x in c['cameras'] if x['name'] == 'left_wrist_camera')[key] = value
            with self.assertRaises(ValueError): validate_cameras(c)
        self.c['camera_system']['capture_interval_steps'] = 5
        with self.assertRaises(ValueError): validate_cameras(self.c)


if __name__ == '__main__': unittest.main()
