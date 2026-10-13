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
