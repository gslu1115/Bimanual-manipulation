import unittest
from dataclasses import replace
import numpy as np
from tests.test_cuboid_pose import CENTRE,SIZE,PRIORS,turn,synthetic_observation,visual_object
from workstation.perception.rgbd_cartons import deproject
from workstation.perception.cuboid_pose import fit_cuboid_pose,grasp_constrained_cuboid_supported,measured_cuboid_supported
from workstation.perception.held_surface import CuboidHoldingTracker
from workstation.skills.visual_pick_place import VisualPickPlace
from workstation.simulation.robot_driver import RobotMotion
from workstation.planning.observed_workspace import ObservedWorkspace


class TapeCarryPoseTests(unittest.TestCase):
    def observation(self,r=None,centre=CENTRE,stripe=True,half_width=.006):
        r=np.eye(3) if r is None else r
        p,s=synthetic_observation(r,centre,views=([0.,-.1001,1.5],))
        cameras=dict(p.cameras);masks={}
        for ref,mask in s.masks.items():
            name=ref.split('/')[0];frame=cameras[name];local=(deproject(frame)-centre)@r
            top=mask&(local[...,2]>SIZE[2]/2-.001)
            rgb=frame.rgb.copy()
            if stripe:rgb[top&(np.abs(local[...,1])<half_width)]=[215,205,170]
            # A smooth lighting gradient alone must not create a tape axis.
            scale=np.clip(1.+local[...,0]*2+local[...,1]*3,.6,1.4)
            rgb=np.clip(rgb.astype(float)*scale[...,None],0,255).astype(np.uint8)
            cameras[name]=replace(frame,rgb=rgb)
            masks[ref]=top&(local[...,0]>-.033)
        return replace(p,cameras=cameras),replace(s,masks=masks)

    def test_tape_measures_direction_on_incomplete_face_instead_of_copying_hint(self):
        for yaw in (0.,28.,-35.):
            r=turn(2,yaw);p,s=self.observation(r)
            result=fit_cuboid_pose(p,s,CENTRE,r@turn(2,3),SIZE,PRIORS,allow_grasp_translation=True)
            self.assertEqual(result['status'],'GRASP_CONSTRAINED',result)
            self.assertEqual(result['observed_face_axes'],[2])
            self.assertTrue(grasp_constrained_cuboid_supported(result,1.))
            self.assertFalse(measured_cuboid_supported(result,1.))
            direction=np.asarray(result['rotation'])[:,0]
            self.assertLess(np.rad2deg(np.arccos(np.clip(direction@r[:,0],-1,1))),1.)
            self.assertEqual(result['predicted_translation_axes'],[0])

    def test_plain_gradient_broad_stripe_or_stale_image_do_not_supply_direction(self):
        for kwargs in ({'stripe':False},{'half_width':.013}):
            p,s=self.observation(**kwargs)
            result=fit_cuboid_pose(p,s,CENTRE,np.eye(3),SIZE,PRIORS,allow_grasp_translation=True)
            self.assertEqual(result['status'],'UNKNOWN',result)
        p,s=self.observation()
        self.assertEqual(fit_cuboid_pose(replace(p,assembled_time_s=1.3),s,CENTRE,np.eye(3),SIZE,PRIORS,
                                       allow_grasp_translation=True)['status'],'UNKNOWN')

    def test_missing_direction_proof_and_overstated_height_are_rejected(self):
        p,s=self.observation()
        result=fit_cuboid_pose(p,s,CENTRE,np.eye(3),SIZE,PRIORS,allow_grasp_translation=True)
        self.assertTrue(grasp_constrained_cuboid_supported(result,1.))
        for changes in ({'tape_direction':None},{'visual_height_lower_bound_m':CENTRE[2]+.1},
                        {'predicted_translation_axes':[2]}):
            self.assertFalse(grasp_constrained_cuboid_supported(dict(result,**changes),1.))

    def test_measured_height_bound_can_prove_lift_without_hidden_xy_becoming_measured(self):
        p,s=self.observation(centre=CENTRE+[0,0,.14])
        obj=visual_object(s,np.eye(3))
        tracker=CuboidHoldingTracker(obj,CENTRE,np.eye(3),SIZE,PRIORS,allow_grasp_translation=True)
        result=tracker.observe(p,s,CENTRE+[0,0,.14],np.eye(3))
        self.assertIsNotNone(result)
        self.assertFalse(result['full_6d_pose_measured'])
        self.assertGreater(result['height_gain_lower_bound_m'],.13)
        self.assertLessEqual(result['height_gain_lower_bound_m'],.14)
        skill=VisualPickPlace.__new__(VisualPickPlace);skill.max_visual_lift=0.
        skill._record_visual_lift(result)
        self.assertEqual(skill.max_visual_lift,result['height_gain_lower_bound_m'])
        skill._record_visual_lift(dict(grasp_constrained=True,height_gain_m=.5))
        self.assertLess(skill.max_visual_lift,.14)

    def test_current_observed_height_does_not_follow_an_unsupported_lift_hint(self):
        p,s=self.observation()
        result=fit_cuboid_pose(p,s,CENTRE+[0,0,.14],np.eye(3),SIZE,PRIORS,allow_grasp_translation=True)
        self.assertEqual(result['status'],'UNKNOWN')

    def test_tape_proof_reaches_motion_gate_without_becoming_full_measurement(self):
        p,s=self.observation();obj=visual_object(s,np.eye(3))
        tracker=CuboidHoldingTracker(obj,CENTRE,np.eye(3),SIZE,PRIORS,allow_grasp_translation=True)
        result=tracker.observe(p,s,CENTRE,np.eye(3))
        driver=RobotMotion.__new__(RobotMotion)
        driver.observed_workspace=ObservedWorkspace(p,PRIORS,replace(s,objects=(obj,)))
        driver.held_visual_geometry=dict(target_id='visual_1',arm='panda_left',offset=tracker.offset.copy(),allow_grasp_translation=True)
        driver.fk=lambda *args:(CENTRE,np.eye(3));driver.monitor=lambda:None;driver.read_states=lambda:p.robot_state
        self.assertTrue(driver.confirm_payload_observation('visual_1',tracker,result,p))
        driver._require_rotating_payload('panda_left','visual_1',SIZE)
        self.assertFalse(driver.observed_workspace.payload_observation['geometric_pose_measured'])
        self.assertFalse(driver.confirm_payload_observation('visual_1',tracker,dict(result,tape_direction=None),p))
        with self.assertRaisesRegex(RuntimeError,'GEOMETRY_UNSUPPORTED'):
            driver._require_rotating_payload('panda_left','visual_1',SIZE)
