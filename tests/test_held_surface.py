import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from workstation.perception.held_surface import HeldSurfaceTracker


class HeldSurfaceTests(unittest.TestCase):
    def setUp(self):
        obj=SimpleNamespace(failures=(),posture='UPRIGHT',visibility='COMPLETE_TOP',
                            position_m=(0.,-.1,.7825),axis_yaw_rad=0.,uncertainty_m=.003)
        self.tracker=HeldSurfaceTracker(obj,np.array(obj.position_m),np.eye(3),(.1,.055,.045))
        x,y=np.meshgrid(np.linspace(-.048,.048,40),np.linspace(-.026,.026,20))
        top=np.stack((x,y,np.full_like(x,.0225)),axis=-1)
        x,z=np.meshgrid(np.linspace(-.048,.048,40),np.linspace(-.021,.019,12))
        side=np.stack((x,np.full_like(x,-.0275),z),axis=-1)
        self.centre=np.array([0.,-.1,.9025])
        self.xyz=np.concatenate((top,side))+self.centre
        shape=self.xyz.shape[:2]
        self.frame=SimpleNamespace(sample_time_s=1.,valid_depth=np.ones(shape,bool))
        self.frame.T_workcell_from_camera_cv=np.eye(4)
        self.frame.T_workcell_from_camera_cv[:3,3]=[0.,-.5,1.5]
        self.packet=SimpleNamespace(observation_time_s=1.,cameras={'scene_camera':self.frame})
        self.scene=SimpleNamespace(masks={'scene_camera/1/partial':np.ones(shape,bool)},
                                   camera_status={'scene_camera':'OK'})

    def observe(self, xyz=None):
        with patch('workstation.perception.held_surface.deproject',return_value=self.xyz if xyz is None else xyz):
            return self.tracker.observe(self.packet,self.scene,self.centre,np.eye(3))

    def test_measured_top_and_side_support_lift_without_full_pose(self):
        result=self.observe()
        self.assertAlmostEqual(result['height_gain_m'],.12)
        self.assertFalse(result['full_6d_pose_measured'])
        self.assertGreater(result['side_points'],20)

    def test_visual_uncertainty_is_retained_and_unbounded_values_rejected(self):
        obj=SimpleNamespace(failures=(),posture='UPRIGHT',visibility='COMPLETE_TOP',
            position_m=(0.,-.1,.7825),axis_yaw_rad=0.,uncertainty_m=.006)
        tracker=HeldSurfaceTracker(obj,np.array(obj.position_m),np.eye(3),(.1,.055,.045))
        self.assertEqual(tracker.uncertainty_m,.006)
        for value in (float('nan'),float('inf'),-.001,.009):
            obj.uncertainty_m=value
            with self.assertRaisesRegex(ValueError,'uncertainty'):
                HeldSurfaceTracker(obj,np.array(obj.position_m),np.eye(3),(.1,.055,.045))

    def test_stationary_box_does_not_match_rising_tcp(self):
        self.assertIsNone(self.observe(self.xyz-[0.,0.,.12]))

    def test_visible_slip_rejected(self):
        self.assertIsNone(self.observe(self.xyz+[.03,0.,0.]))

    def test_tiny_top_patch_not_holding_evidence(self):
        xyz=self.xyz.copy();xyz[:,:,:2]=self.centre[:2]
        self.assertIsNone(self.observe(xyz))

    def test_merged_mask_not_cropped_into_good_inliers(self):
        xyz=self.xyz.copy();xyz[-4:,:,0]+=.15
        self.assertIsNone(self.observe(xyz))

    def test_old_frame_not_relabelled_as_current(self):
        self.frame.sample_time_s=.9
        self.assertIsNone(self.observe())

    def test_unsynchronized_camera_not_used_for_holding(self):
        self.scene.camera_status['scene_camera']='UNSYNCED'
        self.assertIsNone(self.observe())

    def test_collision_translation_comes_from_observed_planes_extents(self):
        xyz=self.xyz+[.003,0.,0.]
        with patch('workstation.perception.held_surface.deproject',return_value=xyz):
            result=self.tracker.collision_estimate(self.packet,self.scene,self.centre,np.eye(3))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['position_m'][0],.003,places=5)
        self.assertEqual(result['top_bottom_semantics'],'UNKNOWN')

    def test_missing_coordinate_constraint_rejects_retreat(self):
        xyz=self.xyz.copy();xyz[:,:,0]*=.4
        with patch('workstation.perception.held_surface.deproject',return_value=xyz):
            self.assertIsNone(self.tracker.collision_estimate(self.packet,self.scene,self.centre,np.eye(3)))

    def test_released_drop_is_measured_from_surfaces_and_does_not_relax_holding(self):
        with patch('workstation.perception.held_surface.deproject',return_value=self.xyz):
            reference=self.tracker.collision_estimate(self.packet,self.scene,self.centre,np.eye(3))
        xyz=self.xyz-[0.,0.,.010]
        with patch('workstation.perception.held_surface.deproject',return_value=xyz):
            self.assertIsNone(self.tracker.observe(self.packet,self.scene,self.centre,np.eye(3)))
            fitted=self.tracker.collision_estimate(self.packet,self.scene,self.centre,np.eye(3),max_drop_m=.013)
        self.assertIsNotNone(fitted)
        np.testing.assert_allclose(fitted['position_m'],np.array(reference['position_m'])-[0.,0.,.010],atol=1e-6)
        self.assertAlmostEqual(fitted['observed_downward_shift_m'],.010)

    def test_release_still_rejects_excess_drop_lateral_slip_and_invalid_bounds(self):
        for shift in ((0.,0.,-.018),(.030,0.,-.010)):
            with patch('workstation.perception.held_surface.deproject',return_value=self.xyz+shift):
                self.assertIsNone(self.tracker.collision_estimate(self.packet,self.scene,self.centre,np.eye(3),max_drop_m=.013))
        for bound in (float('nan'),-.001,.021):
            with self.assertRaisesRegex(ValueError,'drop bound'):
                self.tracker.collision_estimate(self.packet,self.scene,self.centre,np.eye(3),max_drop_m=bound)

    def test_released_consistency_anchor_does_not_follow_post_open_arm_motion(self):
        xyz=self.xyz-[0.,0.,.010]
        with patch('workstation.perception.held_surface.deproject',return_value=xyz):
            fitted=self.tracker.collision_estimate(self.packet,self.scene,self.centre+[0.,0.,.015],np.eye(3),
                max_drop_m=.013,reference_position_m=self.centre)
        self.assertIsNotNone(fitted)
        self.assertAlmostEqual(fitted['observed_downward_shift_m'],.010)
