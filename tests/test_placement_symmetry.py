import math
import unittest
import numpy as np
from workstation.planning.selection import nearest_aligned_yaw, nearest_upright_slot_rotation
from workstation.baselines.handover_math import angle_between
from workstation.observations.camera_geometry import quaternion_from_matrix


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class PlacementSymmetryTests(unittest.TestCase):
    def test_yaw_uses_nearest_equivalent_without_wrapping_a_full_turn(self):
        for degrees, expected in ((179,180), (-179,-180), (359,360),
                                  (181,180), (89,0), (-89,0), (541,540)):
            yaw = math.radians(degrees)
            result = nearest_aligned_yaw(yaw)
            self.assertAlmostEqual(math.degrees(result), expected)
            self.assertLessEqual(abs(result-yaw), math.pi/2)

    def test_full_pose_keeps_top_up_and_uses_closer_axis_equivalent(self):
        tilt = .07
        rx = np.array([[1.,0.,0.], [0.,math.cos(tilt),-math.sin(tilt)],
                       [0.,math.sin(tilt),math.cos(tilt)]])
        for degrees in (-179, 179, 11, 58, 91):
            current = rz(math.radians(degrees)) @ rx
            goal = nearest_upright_slot_rotation(current)
            np.testing.assert_array_equal(goal[:,2], [0.,0.,1.])
            distance = angle_between(quaternion_from_matrix(current), quaternion_from_matrix(goal))
            for alternative in (np.eye(3), rz(math.pi)):
                self.assertLessEqual(distance, angle_between(quaternion_from_matrix(current),
                    quaternion_from_matrix(alternative))+1e-10)

    def test_symmetric_body_hypothesis_keeps_the_same_tcp_goal(self):
        body = rz(math.radians(179))
        tcp = body @ np.diag([1.,-1.,-1.])
        equivalent = np.diag([-1.,-1.,1.])
        first = nearest_upright_slot_rotation(body) @ (tcp.T @ body).T
        second = nearest_upright_slot_rotation(body @ equivalent) @ (tcp.T @ body @ equivalent).T
        np.testing.assert_allclose(first, second, atol=1e-12)
        self.assertLess(angle_between(quaternion_from_matrix(tcp), quaternion_from_matrix(first)), .02)

    def test_invalid_values_do_not_create_a_placement_goal(self):
        for yaw in (float('nan'), float('inf')):
            with self.assertRaises(ValueError): nearest_aligned_yaw(yaw)
        for rotation in (np.full((3,3),float('nan')), np.diag([1.,1.,-1.]), np.ones((3,3))):
            with self.assertRaises(ValueError): nearest_upright_slot_rotation(rotation)
