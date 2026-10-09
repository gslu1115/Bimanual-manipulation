import unittest
import numpy as np
from workstation.planning.robot_geometry import ConvexLink


class ConvexRobotGeometryTests(unittest.TestCase):
    def cube(self):
        normals=np.r_[np.eye(3),-np.eye(3)]
        return ConvexLink(np.column_stack((normals,np.full(6,-.01))),np.full(3,-.01),np.full(3,.01),'fixture')

    def test_robot_contains_only_its_model_not_adjacent_unknown(self):
        m=self.cube();inside=m.contains([[0,0,0],[.02,0,0]],np.zeros(3),np.eye(3),margin=0)
        np.testing.assert_array_equal(inside,[True,False])

    def test_image_time_rotation_and_translation_are_applied(self):
        m=self.cube();angle=.7;c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        p=np.array([.2,-.1,1.]);points=np.array([[0,0,0],[.03,0,0]])@r.T+p
        np.testing.assert_array_equal(m.contains(points,p,r),[True,False])

    def test_known_model_allowance_is_bounded(self):
        m=self.cube()
        np.testing.assert_array_equal(m.contains([[.012,0,0],[.015,0,0]],np.zeros(3),np.eye(3)),[True,False])

    def test_finger_sweep_contains_all_slider_positions_only(self):
        m=self.cube().extruded([0,-.04,0])
        np.testing.assert_array_equal(m.contains([[0,0,0],[0,-.02,0],[0,-.04,0],[0,-.07,0],[.02,-.02,0]],
            np.zeros(3),np.eye(3),margin=0),[True,True,True,False,False])
