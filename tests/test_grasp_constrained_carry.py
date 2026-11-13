"""Partial carry estimates retain a grasp hypothesis without claiming full observation."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from tests.test_cuboid_pose import CENTRE,SIZE,PRIORS,visual_object,synthetic_observation
from workstation.perception.rgbd_cartons import deproject
from workstation.perception.cuboid_pose import fit_cuboid_pose,measured_cuboid_supported,grasp_constrained_cuboid_supported
from workstation.perception.held_surface import CuboidHoldingTracker
from workstation.planning.observed_workspace import ObservedWorkspace
from workstation.simulation.robot_driver import RobotMotion


class GraspConstrainedCarryTests(unittest.TestCase):
    def partial(self):
        p,s=synthetic_observation(np.eye(3),views=([0.,-.45,1.28],))
        masks={ref:mask&(np.abs(deproject(p.cameras[ref.split('/')[0]])[...,0]-CENTRE[0])<.018)
               for ref,mask in s.masks.items()}
        return p,replace(s,masks=masks)

    def fit(self,p,s,**kwargs):
        return fit_cuboid_pose(p,s,CENTRE,np.eye(3),SIZE,PRIORS,**kwargs)

    def test_clipped_parked_carton_encloses_hidden_axis_and_blocks_nearby_lanes(self):
        from workstation.diagnostics.graspgen_pair_place import parked_surface_envelope
        p,s=self.partial()
        obj=visual_object(s,np.eye(3))
        reference=dict(position_m=CENTRE,axis_yaw_rad=0.,size_m=SIZE)
        distant=[(CENTRE[0]+.3,CENTRE[1])]
        bounded,proof=parked_surface_envelope(p,s,reference,[obj],PRIORS,distant)
        self.assertEqual(proof['bounded_axis'],0)
        self.assertGreater(bounded.size_m[0],SIZE[0]+.02)
        self.assertIsNone(bounded.measured_cuboid)
        self.assertIsNone(parked_surface_envelope(p,s,reference,[obj],PRIORS,[CENTRE[:2]]))
        self.assertIsNone(parked_surface_envelope(p,s,reference,[replace(obj,mask_refs=())],PRIORS,distant))
        self.assertIsNone(parked_surface_envelope(p,s,reference,[obj,obj],PRIORS,distant))

    def test_visible_parked_top_requires_current_metric_depth_at_the_reference(self):
        from workstation.diagnostics.graspgen_pair_place import parked_top_still_visible
        p,_=self.partial()
        r=dict(position_m=CENTRE,axis_yaw_rad=0.,size_m=SIZE)
        self.assertTrue(parked_top_still_visible(p,r))
        self.assertFalse(parked_top_still_visible(p,dict(r,position_m=CENTRE+[0,0,.1])))
        self.assertFalse(parked_top_still_visible(replace(p,observation_time_s=2.),r))

    def test_one_side_mask_needs_a_separate_current_top_view(self):
        from workstation.diagnostics.graspgen_pair_place import parked_surface_envelope
        p,s=synthetic_observation(np.eye(3),views=([0.,-.101,1.35],[-.32,-.10,1.08]))
        ref=next(ref for ref in s.masks if not ref.startswith('scene_camera/'))
        frame=p.cameras[ref.split('/')[0]];xyz=deproject(frame)
        mask=s.masks[ref] & (np.abs(xyz[:,:,0]-CENTRE[0]+SIZE[0]/2)<.001) & (xyz[:,:,2]<CENTRE[2]+.015)
        s=replace(s,masks={ref:mask});obj=visual_object(s,np.eye(3))
        reference=dict(position_m=CENTRE,axis_yaw_rad=0.,size_m=SIZE)
        def bound(packet):return parked_surface_envelope(packet,s,reference,[obj],PRIORS,[(.3,-.1)])
        proxy,proof=bound(p)
        self.assertEqual(proof['bounded_axis'],1)
        self.assertIsNone(proxy.measured_cuboid)
        cameras=dict(p.cameras);cameras['scene_camera']=None
        status=dict(p.camera_status);status['scene_camera']='MISSING'
        self.assertIsNone(bound(replace(p,cameras=cameras,camera_status=status)))

    def test_two_fragments_may_independently_measure_the_whole_parked_pose(self):
        from workstation.diagnostics.graspgen_pair_place import parked_surface_envelope
        p,s=synthetic_observation(np.eye(3),views=([0.,-.45,1.28],[-.32,-.10,1.08]))
        reference=dict(position_m=CENTRE,axis_yaw_rad=0.,size_m=SIZE)
        fragments=[replace(visual_object(s,np.eye(3)),track_id=ref,mask_refs=(ref,)) for ref in s.masks]
        proxy,proof=parked_surface_envelope(p,s,reference,fragments,PRIORS,[(.3,-.1)])
        self.assertIsNone(proof['bounded_axis'])
        self.assertEqual(set(proxy.mask_refs),set(s.masks))
        self.assertIsNone(parked_surface_envelope(p,s,reference,fragments,PRIORS,[CENTRE[:2]]))

    def test_occlusion_requires_foreground_in_multiple_views_and_no_contradictory_ray(self):
        from workstation.diagnostics.graspgen_pair_place import parked_top_occluded
        p,_=self.partial()
        frame=next(f for f in p.cameras.values() if f is not None)
        r=dict(position_m=CENTRE,axis_yaw_rad=0.,size_m=SIZE)
        def view(depth):
            return replace(frame,depth_m=np.full_like(frame.depth_m,depth),valid_depth=np.ones_like(frame.valid_depth))
        def packet(depths):
            return SimpleNamespace(observation_time_s=p.observation_time_s,
                cameras={str(i):view(d) for i,d in enumerate(depths)},
                camera_status={str(i):'OK' for i in range(len(depths))})
        self.assertTrue(parked_top_occluded(packet([.05,.05]),r))
        self.assertFalse(parked_top_occluded(packet([.05,4.]),r))
        self.assertFalse(parked_top_occluded(packet([.05]),r))

    def test_partial_translation_is_explicit_and_never_full_measurement(self):
        p,s=self.partial()
        self.assertEqual(self.fit(p,s)['status'],'UNKNOWN')
        result=self.fit(p,s,allow_grasp_translation=True)
        self.assertEqual(result['status'],'GRASP_CONSTRAINED',result)
        self.assertEqual(result['predicted_translation_axes'],[0])
        self.assertTrue(grasp_constrained_cuboid_supported(result,1.))
        self.assertFalse(measured_cuboid_supported(result,1.))
        self.assertFalse(result['geometric_pose_measured'])
        np.testing.assert_allclose(result['position_m'],CENTRE,atol=.003)
        self.assertLess(next(c for c in result['coordinate_constraints'] if c['axis']==0)['observed_span_m'],SIZE[0]/2)

    def test_freshness_and_actual_surface_evidence_are_still_required(self):
        p,s=self.partial()
        self.assertEqual(self.fit(replace(p,assembled_time_s=1.3),s,allow_grasp_translation=True)['status'],'UNKNOWN')
        self.assertEqual(self.fit(p,replace(s,masks={}),allow_grasp_translation=True)['status'],'UNKNOWN')
        self.assertEqual(self.fit(p,replace(s,camera_status={n:'INVALID' for n in s.camera_status}),
                                  allow_grasp_translation=True)['status'],'UNKNOWN')

    def test_one_clipped_face_does_not_supply_rotation_or_two_translation_axes(self):
        p,s=self.partial()
        masks={ref:mask&(deproject(p.cameras[ref.split('/')[0]])[...,2]>CENTRE[2]+SIZE[2]/2-.001)
               for ref,mask in s.masks.items()}
        self.assertEqual(self.fit(p,replace(s,masks=masks),allow_grasp_translation=True)['status'],'UNKNOWN')

    def test_bad_partial_proof_and_predicted_height_are_rejected(self):
        p,s=self.partial();result=self.fit(p,s,allow_grasp_translation=True)
        for changes in ({'geometric_pose_measured':True},{'coordinate_constraints':[]},
                        {'predicted_translation_axes':[2]},{'predicted_translation_axes':[0,1]},
                        {'observed_face_axes':[2]},{'time_s':.5}):
            with self.subTest(changes=changes):
                self.assertFalse(grasp_constrained_cuboid_supported(dict(result,**changes),1.))

    def test_tracker_requires_opt_in_and_preserves_original_relation(self):
        p,s=self.partial();obj=visual_object(s,np.eye(3))
        strict=CuboidHoldingTracker(obj,CENTRE,np.eye(3),SIZE,PRIORS)
        carry=CuboidHoldingTracker(obj,CENTRE,np.eye(3),SIZE,PRIORS,allow_grasp_translation=True)
        self.assertIsNone(strict.observe(p,s,CENTRE,np.eye(3)))
        result=carry.observe(p,s,CENTRE,np.eye(3))
        self.assertIsNotNone(result)
        self.assertFalse(result['full_6d_pose_measured'])
        np.testing.assert_array_equal(carry.offset,np.zeros(3))

    def test_driver_and_workspace_keep_partial_provenance_and_revoke_bad_evidence(self):
        p,s=self.partial();obj=visual_object(s,np.eye(3))
        tracker=CuboidHoldingTracker(obj,CENTRE,np.eye(3),SIZE,PRIORS,allow_grasp_translation=True)
        result=tracker.observe(p,s,CENTRE,np.eye(3))
        driver=RobotMotion.__new__(RobotMotion)
        driver.observed_workspace=ObservedWorkspace(p,PRIORS,replace(s,objects=(obj,)))
        driver.held_visual_geometry=dict(target_id='visual_1',arm='panda_left',offset=tracker.offset.copy(),
                                        allow_grasp_translation=True)
        driver.fk=lambda arm,q:(CENTRE,np.eye(3))
        driver.monitor=lambda:None
        driver.read_states=lambda:p.robot_state
        self.assertTrue(driver.confirm_payload_observation('visual_1',tracker,result,p))
        payload=driver.observed_workspace.payload_observation
        self.assertTrue(payload['grasp_constrained']);self.assertFalse(payload['geometric_pose_measured'])
        self.assertIsNone(driver.held_visual_geometry['measured_pose_time_s'])
        driver._require_rotating_payload('panda_left','visual_1',SIZE)
        driver.held_visual_geometry['allow_grasp_translation']=False
        with self.assertRaises(RuntimeError):driver._require_rotating_payload('panda_left','visual_1',SIZE)
        driver.held_visual_geometry['allow_grasp_translation']=True
        self.assertFalse(driver.confirm_payload_observation('visual_1',tracker,dict(result,coordinate_constraints=[]),p))
        self.assertIsNone(driver.observed_workspace.payload_observation)
