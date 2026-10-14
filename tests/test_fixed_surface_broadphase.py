import unittest
import numpy as np
from tests import test_observed_workspace as fixtures
from workstation.planning.collision import AABB
from workstation.planning.occlusion_memory import OcclusionMemory
from workstation.perception.rgbd_cartons import deproject


class FixedSurfaceBroadphaseTests(unittest.TestCase):
    def test_exact_original_surface_pixels_and_valid_depth(self):
        rng=np.random.default_rng(703)
        for depth in (.998,.999,1.,1.001,1.002,.7):
            ws=fixtures.WorkspaceTests().workspace(depth=depth)
            frame=next(ws._frames());frame.depth_m[5:10,6:12]=.6
            frame.valid_depth[rng.random(frame.depth_m.shape)<.1]=False
            ws.static=(*ws.static,AABB('small',(.02,0,.86),(.02,.03,.03)),AABB('far',(4,3,2),(1,1,1)))
            xyz=deproject(frame);expected=np.zeros(frame.depth_m.shape,bool)
            for box in ws.static:
                distance=np.abs(xyz-np.asarray(box.centre))-np.asarray(box.size)/2
                expected|=(np.max(distance,axis=-1)<=.002)&(np.min(np.abs(distance),axis=-1)<=.002)
            expected&=frame.valid_depth
            with self.subTest(depth=depth):
                np.testing.assert_array_equal(OcclusionMemory._fixed_surface_depth_mask(ws,frame),expected)
