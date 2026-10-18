import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from tests.test_cuboid_pose import synthetic_observation,CENTRE,SIZE
from workstation.perception.rgbd_cartons import deproject,CartonEstimator
from workstation.perception.held_surface import HeldSurfaceTracker,released_bound_supported
from workstation.perception.scene_estimate import pose_is_observed,ScenePriors
from workstation.planning.collision import scene_obstacles


class ReleasedBoundsTests(unittest.TestCase):
    def setUp(self):
        self.packet,self.scene=synthetic_observation(np.eye(3),views=([0.,-.1001,1.35],))
        ref=next(iter(self.scene.masks));xyz=deproject(self.packet.cameras[ref.split('/')[0]])
        self.scene=replace(self.scene,masks={ref:self.scene.masks[ref] & (xyz[...,0]>CENTRE[0]-.041)})
        obj=SimpleNamespace(failures=(),posture='UPRIGHT',visibility='COMPLETE_TOP',
            position_m=CENTRE,axis_yaw_rad=0.,uncertainty_m=.003)
        self.tracker=HeldSurfaceTracker(obj,CENTRE,np.eye(3),SIZE)
        self.reference=CENTRE+[0,0,.012]

    def fit(self,packet=None,scene=None,reference=None):
        return self.tracker.collision_estimate(packet or self.packet,scene or self.scene,
            self.reference,np.eye(3),max_drop_m=.016,reference_position_m=self.reference if reference is None else reference,
            reference_rotation=np.eye(3),allow_partial=True)

    def test_top_only_is_a_collision_envelope_and_not_a_pose_or_hold(self):
        self.assertIsNone(self.tracker.collision_estimate(self.packet,self.scene,self.reference,np.eye(3),
            max_drop_m=.016,reference_position_m=self.reference))
        result=self.fit();self.assertIsNotNone(result)
        self.assertTrue(released_bound_supported(result,1.));self.assertEqual(result['side_points'],0)
        self.assertFalse(result['geometric_pose_measured']);self.assertFalse(result['grasp_constrained'])
        bounds=np.asarray(result['centre_intervals_local_m'])
        self.assertTrue(np.all(bounds[:,0]<=CENTRE+1e-5));self.assertTrue(np.all(bounds[:,1]>=CENTRE-1e-5))
        self.assertGreater(result['collision_size_m'][0],SIZE[0]+.007)

    def test_interval_centre_comes_from_current_patch_not_old_hint(self):
        a=self.fit(reference=self.reference+[.002,0,0]);b=self.fit(reference=self.reference-[.002,0,0])
        np.testing.assert_allclose(a['position_m'],b['position_m'],atol=1e-8)
        np.testing.assert_allclose(a['centre_intervals_local_m'],b['centre_intervals_local_m'],atol=1e-8)

    def test_released_body_does_not_follow_retracting_tcp(self):
        result=self.tracker.collision_estimate(self.packet,self.scene,self.reference+[.1,0,.15],np.diag([-1.,-1.,1.]),
            max_drop_m=.016,reference_position_m=self.reference,reference_rotation=np.eye(3),allow_partial=True)
        np.testing.assert_allclose(result['position_m'],self.fit()['position_m'],atol=1e-8)

    def test_tiny_patch_stale_or_missing_frames_remain_rejected(self):
        ref=next(iter(self.scene.masks));name=ref.split('/')[0];frame=self.packet.cameras[name]
        xyz=deproject(frame);tiny=replace(self.scene,masks={ref:self.scene.masks[ref]&(np.abs(xyz[...,0]-CENTRE[0])<.01)})
        self.assertIsNone(self.fit(scene=tiny))
        self.assertIsNone(self.fit(packet=replace(self.packet,assembled_time_s=1.3)))
        old=replace(frame,sample_time_s=.8,robot_state_at_frame={name:replace(state,sample_time_s=.8)
            for name,state in frame.robot_state_at_frame.items()})
        self.assertIsNone(self.fit(packet=replace(self.packet,cameras={**self.packet.cameras,name:old})))
        self.assertIsNone(self.fit(scene=replace(self.scene,masks={})))

    def test_wrong_location_drop_and_contaminated_mask_remain_rejected(self):
        self.assertIsNone(self.fit(reference=self.reference+[.04,0,0]))
        self.assertIsNone(self.fit(reference=self.reference+[0,0,.03]))
        ref=next(iter(self.scene.masks));mask=self.scene.masks[ref].copy();mask[:50,:50]=True
        self.assertIsNone(self.fit(scene=replace(self.scene,masks={ref:mask})))

    def test_envelope_not_final_pose_and_does_not_shrink_collision_volume(self):
        from tests.test_scene_estimate import packet
        est=CartonEstimator(ScenePriors(tuple(SIZE),.76,(-.38,.38,-.51,.16)))
        template=est.estimate(packet(yaw=0.)).objects[0];result=self.fit()
        size=result['collision_size_m']
        obj=replace(template,position_m=result['position_m'],axis_yaw_rad=result['axis_yaw_rad'],
            size_m=tuple(SIZE),visibility='RELEASE_BOUNDED_CUBOID',posture='UNKNOWN',failures=(),
            measured_cuboid=None,uncertainty_m=.003,quality=dict(geometric_length_m=size[0],geometric_width_m=size[1],observed_height_m=size[2]))
        scene=replace(self.scene,objects=(obj,))
        self.assertFalse(pose_is_observed(obj,scene))
        obstacle=scene_obstacles(scene,None,extra_margin=0.)[0]
        self.assertGreaterEqual(obstacle.size[0],SIZE[0]+.007)
        self.assertFalse(released_bound_supported(dict(result,time_s=.8),1.))
        self.assertFalse(released_bound_supported(dict(result,collision_size_m=[.01,.01,.01]),1.))


if __name__=='__main__':unittest.main()
