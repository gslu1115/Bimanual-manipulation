import unittest
from dataclasses import replace
import numpy as np
from workstation.planning.graspgen_candidates import candidates_from_result,opening_from_visual_box,upright_body_rotation
from workstation.planning.selection import assess
from workstation.observations.camera_geometry import pose_matrix
from workstation.simulation.robot_driver import RobotMotion
from tests import test_visual_planning


class GraspGenPlanningTests(unittest.TestCase):
    def setUp(self):
        fixture=test_visual_planning.PlanningTests();fixture.setUp()
        self.scene,self.states,self.obj=fixture.scene,fixture.states,fixture.scene.objects[0]
        self.calibration={'verified':True,'T_graspgen_base_from_project_tcp':np.eye(4).tolist(),
                          'gripper_geometry_equivalence_verified':False}
        self.pose=np.eye(4);self.pose[:3,:3]=upright_body_rotation(self.obj)@np.diag([1.,-1.,-1.])
        self.pose[:3,3]=self.obj.position_m
        self.result=dict(base_poses_workcell=[self.pose],scores=[.91],request_id='test',
            observation_time_s=self.obj.time_s,metadata={'gripper_name':'franka_panda'})

    def test_full_rotation_and_provenance_reach_both_arms_without_projection(self):
        a=.08;c,s=np.cos(a),np.sin(a)
        self.pose[:3,:3]=self.pose[:3,:3]@np.array([[c,0,s],[0,1,0],[-s,0,c]])
        items=candidates_from_result(self.result,self.obj,self.states,self.calibration,{'scene_camera':17})
        self.assertEqual(len(items),2)
        for item in items:
            np.testing.assert_allclose(pose_matrix(item.position_m,item.orientation_wxyz),self.pose,atol=1e-12)
            self.assertEqual(item.metadata['sequence_ids']['scene_camera'],17)
            self.assertFalse(item.metadata['model_gripper_geometry_equivalent'])
        seen=[]
        rows=assess(self.scene,self.states,lambda c,o,s:seen.append(c) or
            dict(reasons=(),joint_cost=0.,path_clearance_m=.03),np.zeros(9),
            candidate_provider=lambda o,states:items)
        self.assertEqual(len(seen),2);self.assertTrue(all(x.feasible for x in rows))
        self.assertTrue(all(x.candidate.source=='graspgen' for x in rows))

    def test_width_is_visual_candidate_specific_and_never_recentres(self):
        width,close,_,reasons=opening_from_visual_box(self.pose,self.obj)
        larger=replace(self.obj,size_m=(.1,.095,.045))
        wide,_,_,reject=opening_from_visual_box(self.pose,larger)
        self.assertLess(width,wide);self.assertLess(close,width)
        self.assertIn('GRIPPER_SPACE_INSUFFICIENT',reject)
        shifted=self.pose.copy();shifted[:3,3]+=shifted[:3,1]*.01
        _,_,_,reject=opening_from_visual_box(shifted,self.obj)
        self.assertIn('GRASP_OFF_CENTRE',reject)
        np.testing.assert_array_equal(self.pose[:3,3],self.obj.position_m)

    def test_stale_wrong_service_and_uncalibrated_candidates_cannot_reach_ik(self):
        for bad in ({**self.result,'observation_time_s':self.obj.time_s-1},
                    {**self.result,'metadata':{'gripper_name':'robotiq'}}):
            with self.assertRaises(ValueError):candidates_from_result(bad,self.obj,self.states,self.calibration,{})
        with self.assertRaises(ValueError):
            candidates_from_result(self.result,self.obj,self.states,{**self.calibration,'verified':False},{})
        items=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})
        items=[replace(c,observation_time_s=c.observation_time_s-1) for c in items]
        def forbidden(*args):self.fail('Stale candidate reached planning')
        rows=assess(self.scene,self.states,forbidden,np.zeros(9),candidate_provider=lambda o,s:items)
        self.assertTrue(all('STALE_CANDIDATE' in r.reasons for r in rows))

    def test_pose_preflight_uses_original_attitude_and_restores_holding_state(self):
        item=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})[0]
        driver=RobotMotion.__new__(RobotMotion)
        driver.read_states=lambda:self.states;driver.home=np.zeros(9);driver.place_support_z=.76
        sentinel={'old':'actual holding evidence'};driver.held_visual_geometry=sentinel
        calls=[]
        def solve(*args,**kw):
            np.testing.assert_allclose(kw['orientation_wxyz'],item.orientation_wxyz);return np.zeros(7)
        driver.solve_pregrasp=solve
        def joint(*args,**kw):driver.last_plan_check={};return [np.zeros(7),np.zeros(7)],.1
        driver.checked_joint_plan=joint
        def cartesian(*args,**kw):
            calls.append((np.array(args[1]),np.array(args[2]),kw))
            return [np.zeros(7),np.zeros(7)]
        driver.cartesian_knots=cartesian
        driver.validate_path=lambda *args,**kw:.1
        result=driver.pose_feasibility(item,self.obj,self.scene,(.3,0.))
        self.assertFalse(result['reasons']);self.assertIs(driver.held_visual_geometry,sentinel)
        self.assertIn('OPEN',result['checked_phases']);self.assertIn('CLOSE',result['checked_phases'])
        np.testing.assert_allclose(calls[0][2]['orientation1'],item.orientation_wxyz)
        np.testing.assert_allclose(calls[0][0],np.asarray(item.position_m)-np.asarray(item.approach_direction)*.18)
        self.assertFalse(result['candidate_execution_authorized'])
        self.assertIn('NOT a holding measurement',result['nominal_visual_payload_relation']['source'])

    def test_preflight_collision_is_not_converted_into_success(self):
        item=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})[0]
        driver=RobotMotion.__new__(RobotMotion);driver.read_states=lambda:self.states
        driver.held_visual_geometry=None
        def blocked(*args,**kw):raise RuntimeError('TARGET_BODY_COLLISION')
        driver.solve_pregrasp=blocked
        report=driver.pose_feasibility(item,self.obj,self.scene)
        self.assertEqual(report['reasons'],('TARGET_BODY_COLLISION',))
        self.assertEqual(report['rejection_phase'],'PREGRASP')
        self.assertIsNone(driver.held_visual_geometry)
