import unittest
from dataclasses import replace
import numpy as np
from workstation.planning.graspgen_candidates import candidates_from_result,opening_from_visual_box,upright_body_rotation
from workstation.planning.selection import assess,visual_pinch_balance
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

    def test_upright_ranking_prefers_more_lift_clearance_among_feasible_grasps(self):
        c=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})[0]
        tight=replace(c,candidate_id='tight',model_score=.99)
        clear=replace(c,candidate_id='clear',model_score=.90)
        def forecast(candidate,obj,scene):
            return dict(reasons=(),joint_cost=0.,path_clearance_m=.02,
                        lift_palm_clearance_m=.003 if candidate is tight else .010)
        rows=assess(self.scene,self.states,forecast,np.zeros(9),
                    candidate_provider=lambda *_:[tight,clear])
        self.assertTrue(all(row.feasible for row in rows))
        self.assertEqual(rows[0].candidate.candidate_id,'clear')
        self.assertEqual(rows[0].terms['lift_palm_cost'],0.)

    def test_contact_margin_ranks_oblique_corner_pinch_below_parallel_sides(self):
        from workstation.planning.selection import upright_contact_cost,distinct_pinches
        c=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})[0]
        width=min(self.obj.size_m[:2])
        def contact(span,margin):
            return replace(c,metadata=dict(opening=dict(estimated_tip_engagement_m=.02,
                projected_span_m=span,available_clearance_per_finger_m=margin)))
        good=contact(width+.001,.006)
        bad=contact(width+.008,.0025)
        self.assertLess(upright_contact_cost(good,self.obj),upright_contact_cost(bad,self.obj))
        different=replace(c,position_m=tuple(np.asarray(c.position_m)+[.008,0,0]))
        self.assertEqual(distinct_pinches([good,bad,different],2),[good,different])

    def test_initial_lift_distinguishes_loading_from_tracking(self):
        from workstation.planning.selection import lift_slip_evidence
        def check(gain):
            return lift_slip_evidence(.80,.86,dict(height_gain_m=gain,uncertainty_m=.004),
                                      [.027,.027],[.028,.028])
        self.assertFalse(check(.057)['suspect'])
        self.assertTrue(check(.049)['suspect'])
        self.assertAlmostEqual(check(.049)['gap_change_m'],-.002)

    def test_shallow_seating_moves_only_along_model_approach(self):
        from workstation.diagnostics.graspgen_pair import refine_pinch_contact
        self.calibration['fingertip_reach_past_tcp_m']=.01
        self.pose[2,3]=self.obj.position_m[2]+self.obj.size_m[2]/2+.003
        c=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})[0]
        seated=refine_pinch_contact(c,self.obj)
        self.assertEqual(seated.orientation_wxyz,c.orientation_wxyz)
        self.assertAlmostEqual(seated.metadata['opening']['estimated_tip_engagement_m'],self.obj.size_m[2]*.40)
        np.testing.assert_allclose(np.asarray(seated.position_m)-c.position_m,
            np.asarray(c.approach_direction)*seated.metadata['axial_seating_m'])
        self.assertEqual(seated.metadata['T_workcell_from_graspgen_base'],c.metadata['T_workcell_from_graspgen_base'])
        blocked=replace(c,rejection_reasons=('UNSUPPORTED_UPRIGHT_APPROACH',))
        self.assertIs(refine_pinch_contact(blocked,self.obj),blocked)
        tangent=self.pose[:3,0]
        shifted=replace(c,position_m=tuple(np.asarray(c.position_m)+.028*tangent))
        balanced=refine_pinch_contact(shifted,self.obj)
        self.assertAlmostEqual(balanced.metadata['tangent_slide_m'],-.020)
        self.assertEqual(shifted.orientation_wxyz,balanced.orientation_wxyz)

    def test_nearby_wide_box_pinch_preserves_approach_and_rechecks_opening(self):
        from workstation.diagnostics.graspgen_pair import refine_pinch_contact
        self.calibration['fingertip_reach_past_tcp_m']=.012
        obj=replace(self.obj,size_m=(.12,.065,.055))
        c,s=np.cos(.15),np.sin(.15)
        self.pose[:3,:3]=self.pose[:3,:3]@np.array([[c,-s,0],[s,c,0],[0,0,1]])
        self.pose[:3,3]+=self.pose[:3,1]*.006
        candidate=candidates_from_result(self.result,obj,self.states,self.calibration,{})[0]
        self.assertIn('GRIPPER_SPACE_INSUFFICIENT',candidate.rejection_reasons)
        adjusted=refine_pinch_contact(candidate,obj)
        self.assertFalse(adjusted.rejection_reasons)
        self.assertLessEqual(adjusted.opening_m,.08)
        self.assertAlmostEqual(abs(adjusted.metadata['pad_alignment_rad']),.15)
        np.testing.assert_allclose(pose_matrix(adjusted.position_m,adjusted.orientation_wxyz)[:3,2],
                                   candidate.approach_direction,atol=1e-8)

    def test_wrist_view_change_renews_pose_only_for_one_stationary_carton(self):
        from workstation.diagnostics.graspgen_pair import renew_view_candidate
        c=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})[0]
        t=self.obj.time_s+.1
        obj=replace(self.obj,track_id='new',time_s=t,velocity_m_s=(0.,0.,0.))
        scene=replace(self.scene,objects=(obj,),observation_time_s=t,assembled_time_s=t)
        refreshed=renew_view_candidate(c,scene,{'scene_camera':18})
        self.assertEqual((refreshed.target_id,refreshed.validated_time_s),('new',t))
        self.assertEqual((refreshed.position_m,refreshed.candidate_id),(c.position_m,c.candidate_id))
        with self.assertRaisesRegex(RuntimeError,'PAIR_TARGET_LOST'):
            renew_view_candidate(c,replace(scene,objects=(obj,replace(obj,track_id='other'))),{})
        with self.assertRaisesRegex(RuntimeError,'MODEL_TARGET_CHANGED'):
            renew_view_candidate(c,replace(scene,objects=(replace(obj,velocity_m_s=(.1,0,0)),)),{})

    def test_full_opening_retains_uncertainty_and_minimum_clearance(self):
        full=replace(self.obj,posture='UPRIGHT',visibility='COMPLETE_TOP',
                     size_m=(.12,.070,.055),uncertainty_m=.003)
        width,_,evidence,reasons=opening_from_visual_box(self.pose,full)
        self.assertEqual(width,.080)
        self.assertGreaterEqual(evidence['available_clearance_per_finger_m'],.0015)
        self.assertFalse(reasons)
        for obj in (replace(full,visibility='PARTIAL_OR_UNCERTAIN'),replace(full,size_m=(.12,.073,.055))):
            self.assertIn('GRIPPER_SPACE_INSUFFICIENT',opening_from_visual_box(self.pose,obj)[3])

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
        self.obj=replace(self.obj,axis_yaw_rad=np.deg2rad(179.))
        self.pose[:3,:3]=upright_body_rotation(self.obj)@np.diag([1.,-1.,-1.])
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
        expected=np.diag([-1.,-1.,1.])
        np.testing.assert_array_equal(result['placement_body_rotation'],expected)
        relation=np.asarray(result['nominal_visual_payload_relation']['relative_rotation'])
        for _,_,call in calls[2:]:
            actual=pose_matrix([0.,0.,0.],call['orientation1'])[:3,:3]
            np.testing.assert_allclose(actual@relation,expected,atol=1e-12)
        np.testing.assert_allclose(pose_matrix(item.position_m,item.orientation_wxyz),self.pose,atol=1e-12)

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

    def test_balance_prefers_central_original_pose_despite_higher_edge_model_score(self):
        centred=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})[0]
        edge=replace(centred,position_m=tuple(np.asarray(centred.position_m)+self.pose[:3,0]*.025),
                     model_score=.99,candidate_id='edge')
        centred=replace(centred,model_score=.3,candidate_id='centred')
        poses=[(c.position_m,c.orientation_wxyz) for c in (edge,centred)]
        rows=assess(self.scene,self.states,lambda c,o,s:dict(reasons=(),joint_cost=0.,path_clearance_m=.03),
                    np.zeros(9),candidate_provider=lambda o,s:[edge,centred])
        self.assertTrue(all(row.feasible for row in rows))
        self.assertEqual(rows[0].candidate.candidate_id,'centred')
        self.assertGreater(rows[1].terms['visual_pinch_balance']['penalty'],.4)
        self.assertEqual(poses,[(c.position_m,c.orientation_wxyz) for c in (edge,centred)])

    def test_balance_is_frame_invariant_and_does_not_bypass_path_rejection(self):
        from workstation.observations.camera_geometry import quaternion_from_matrix
        candidate=candidates_from_result(self.result,self.obj,self.states,self.calibration,{})[0]
        edge=replace(candidate,position_m=tuple(np.asarray(candidate.position_m)+self.pose[:3,0]*.02))
        original=visual_pinch_balance(edge,self.obj)
        a=.7;c,s=np.cos(a),np.sin(a);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        t=np.array([.3,-.7,.1]);obj=replace(self.obj,position_m=tuple(r@self.obj.position_m+t),
                                         axis_yaw_rad=self.obj.axis_yaw_rad+a)
        moved=replace(edge,position_m=tuple(r@edge.position_m+t),
                      orientation_wxyz=tuple(quaternion_from_matrix(r@self.pose[:3,:3])))
        self.assertAlmostEqual(visual_pinch_balance(moved,obj)['penalty'],original['penalty'])
        rows=assess(self.scene,self.states,lambda c,o,s:dict(reasons=('SELF_COLLISION',),
                    joint_cost=0.,path_clearance_m=0.),np.zeros(9),candidate_provider=lambda o,s:[candidate])
        self.assertFalse(rows[0].feasible);self.assertIn('SELF_COLLISION',rows[0].reasons)
        self.assertEqual(visual_pinch_balance(replace(candidate,source='geometry'),self.obj)['penalty'],0.)
