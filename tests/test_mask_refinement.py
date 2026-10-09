import unittest
from dataclasses import replace
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import deproject
from workstation.perception.mask_refinement import refine_visible_boundary


class MaskRefinementTests(unittest.TestCase):
    def setUp(self):
        self.frame=packet(yaw=0.,tape=False).cameras['scene_camera']
        self.truth=self.frame.depth_m < .7
        self.seed=self.truth.copy()
        v,u=np.nonzero(self.truth);self.left=int(u.min())
        self.missing=self.truth & (np.indices(self.truth.shape)[1] < self.left+3)
        self.seed[self.missing]=False

    def test_visible_same_face_omission_restored(self):
        refined=refine_visible_boundary(self.frame,self.seed,deproject(self.frame))
        self.assertTrue(refined[self.missing].all())
        self.assertFalse(refined[~self.truth].any())

    def test_invalid_depth_never_filled(self):
        valid=self.frame.valid_depth.copy();valid[self.missing]=False
        frame=replace(self.frame,valid_depth=valid)
        self.assertFalse(refine_visible_boundary(frame,self.seed,deproject(frame))[self.missing].any())

    def test_robot_colour_on_same_plane_not_added(self):
        rgb=self.frame.rgb.copy();rgb[self.missing]=[114,99,84]
        frame=replace(self.frame,rgb=rgb)
        self.assertFalse(refine_visible_boundary(frame,self.seed,deproject(frame))[self.missing].any())

    def test_same_colour_at_different_depth_not_added(self):
        depth=self.frame.depth_m.copy();depth[self.missing]+=.015
        frame=replace(self.frame,depth_m=depth)
        self.assertFalse(refine_visible_boundary(frame,self.seed,deproject(frame))[self.missing].any())

    def test_other_instance_mask_not_invaded(self):
        refined=refine_visible_boundary(self.frame,self.seed,deproject(self.frame),blocked=self.missing)
        self.assertFalse(refined[self.missing].any())
