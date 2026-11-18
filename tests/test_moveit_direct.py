import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from workstation.planning import moveit_pair


class DirectApproachTests(unittest.TestCase):
    def run_request(self, checks):
        robot = SimpleNamespace(moveit_urdf='robot', moveit_acceleration=[], moveit_socket='socket')
        payload = dict(frame_ids={'scene_camera': 1}, occupied=[])
        modes = []

        def plan(address, request):
            direct = request['prefer_direct']
            modes.append(direct)
            result = dict(route='direct' if direct else 'rrt_connect', search_s=0., sampler_calls=0)
            return result, np.zeros((2, 2, 7)), np.array([0., 1.])

        with patch.object(moveit_pair, 'request', return_value=payload), \
             patch.object(moveit_pair, '_timed_approach', side_effect=plan), \
             patch.object(moveit_pair, '_check_waypoints', side_effect=checks):
            result = moveit_pair.checked_approach(robot, None, None, None, None)
        return modes, result[-1]

    def test_clear_direct_path_requires_one_request(self):
        modes, details = self.run_request([(.012, 2)])
        self.assertEqual(modes, [True])
        self.assertEqual(details['route'], 'direct')

    def test_project_collision_retries_solver_and_checks_returned_path(self):
        modes, details = self.run_request([RuntimeError('OTHER_ARM_COLLISION'), (.012, 2)])
        self.assertEqual(modes, [True, False])
        self.assertEqual(details['route'], 'rrt_connect')
        self.assertEqual(details['direct_project_rejection'], 'OTHER_ARM_COLLISION')

    def test_second_collision_and_stale_measurement_propagate(self):
        for errors, expected in [
            (['OTHER_ARM_COLLISION', 'SELF_COLLISION'], 'SELF_COLLISION'),
            (['STALE_OBSERVATION'], 'STALE_OBSERVATION'),
        ]:
            with self.subTest(errors=errors), self.assertRaisesRegex(RuntimeError, expected):
                self.run_request([RuntimeError(msg) for msg in errors])
