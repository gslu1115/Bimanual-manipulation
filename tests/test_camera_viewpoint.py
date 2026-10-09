import unittest
import numpy as np
from workstation.planning.camera_viewpoint import rigid_transform,camera_goal_to_tcp


class CameraViewpointTests(unittest.TestCase):
    def test_image_time_mount_and_tcp_are_preserved(self):
        angle=.8;c,s=np.cos(angle),np.sin(angle)
        hand=rigid_transform([.3,-.2,1.2],[[c,-s,0],[s,c,0],[0,0,1]])
        mount=rigid_transform([.065,.06,.025],np.diag([1,-1,-1]))
        tool=rigid_transform([0,0,.1],np.eye(3));camera=hand@mount;tcp=hand@tool
        desired=rigid_transform([-.1,.12,1.04],np.eye(3))
        result,estimated=camera_goal_to_tcp(hand,tcp,camera,desired)
        np.testing.assert_allclose(estimated,mount,atol=1e-12)
        np.testing.assert_allclose(result@np.linalg.inv(tool)@mount,desired,atol=1e-12)

    def test_nonfinite_calibration_is_rejected(self):
        bad=np.eye(4);bad[0,0]=float('nan')
        with self.assertRaises(ValueError):camera_goal_to_tcp(bad,np.eye(4),np.eye(4),np.eye(4))
