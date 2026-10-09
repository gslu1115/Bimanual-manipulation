import unittest
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.models.graspgen_client import visual_clouds,validate_grasps,tcp_poses,GraspGenClient


class GraspGenInterfaceTests(unittest.TestCase):
    def test_observed_cloud_is_metric_workcell_only(self):
        p=packet(); scene=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))).estimate(p)
        clouds=visual_clouds(p,scene,scene.objects[0].track_id)
        self.assertEqual(clouds.frame,'workcell'); self.assertFalse(clouds.unseen_space_is_free)
        self.assertAlmostEqual(float(clouds.target_m[:,2].mean()),.805,places=4)
        self.assertEqual(clouds.target_m.shape[1],3)

    def test_tcp_calibration_required_and_composed_once(self):
        grasp=np.eye(4)[None]; grasp[0,:3,3]=[.1,.2,.3]
        transform=np.eye(4); transform[2,3]=.1
        with self.assertRaises(ValueError): tcp_poses(grasp,transform)
        np.testing.assert_allclose(tcp_poses(grasp,transform,True)[0,:3,3],[.1,.2,.4])

    def test_invalid_grasp_rotation_rejected(self):
        bad=np.eye(4)[None]; bad[0,0,0]=2
        with self.assertRaises(ValueError): validate_grasps(bad,[.8])

    def test_wrong_gripper_rejected(self):
        client=GraspGenClient(); client.request=lambda unused:{'gripper_name':'robotiq_2f_140'}
        with self.assertRaises(ValueError): client.metadata()
