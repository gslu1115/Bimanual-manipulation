import gc
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from workstation.observations.observation_packet import CameraFrame, RobotState, ROBOT_NAMES
from workstation.observations.camera_geometry import deproject_depth
from workstation.perception.rgbd_cartons import deproject


class FramePointReuseTests(unittest.TestCase):
    def frame(self):
        transform=np.eye(4)
        transform[:3,:3]=[[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]]
        transform[:3,3]=[.3,-.5,.8]
        return CameraFrame('scene_camera',np.zeros((2,3,3),np.uint8),
            np.array([[1.,np.inf,2.],[3.,0.,4.]],np.float32),
            np.array([[True,False,True],[True,False,True]]),
            np.array([[100.,0.,1.],[0.,90.,.5],[0.,0.,1.]]),transform,1.,7,
            {name:RobotState(1.,np.zeros(7),np.full(2,.04)) for name in ROBOT_NAMES})

    def test_shared_read_only_cloud_preserves_exact_projection(self):
        frame=self.frame()
        expected=deproject_depth(frame)
        with patch('workstation.observations.camera_geometry.deproject_depth',wraps=deproject_depth) as compute:
            first=deproject(frame);second=deproject(frame)
        compute.assert_called_once_with(frame)
        self.assertIs(first,second)
        np.testing.assert_array_equal(first,expected)
        with self.assertRaises(ValueError):first[0,0,0]=99.

    def test_replaced_frame_with_same_sequence_recomputes_depth_and_pose(self):
        frame=self.frame();old=deproject(frame)
        depth=frame.depth_m.copy();depth[0,0]=.5
        transform=frame.T_workcell_from_camera_cv.copy();transform[0,3]+=.2
        new=replace(frame,depth_m=depth,T_workcell_from_camera_cv=transform)
        measured=deproject(new)
        self.assertIsNot(measured,old)
        np.testing.assert_array_equal(measured,deproject_depth(new))
        self.assertFalse(np.array_equal(old,measured))
        depth[0,0]=20.;transform[0,3]=30.
        np.testing.assert_array_equal(deproject(new),measured)

    def test_mutable_fixture_never_uses_snapshot_cache(self):
        frame=self.frame()
        legacy=SimpleNamespace(depth_m=frame.depth_m.copy(),valid_depth=frame.valid_depth.copy(),
            K=frame.K.copy(),T_workcell_from_camera_cv=frame.T_workcell_from_camera_cv.copy())
        before=deproject(legacy);legacy.depth_m[0,0]=.5
        after=deproject(legacy)
        self.assertFalse(np.array_equal(before,after))
        np.testing.assert_array_equal(after,deproject_depth(legacy))

    def test_cache_lifetime_is_owned_by_frame_without_global_retention(self):
        frame=self.frame();points=deproject(frame)
        reference=weakref.ref(points)
        del points;del frame;gc.collect()
        self.assertIsNone(reference())
