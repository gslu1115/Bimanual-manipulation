import unittest
from types import SimpleNamespace
import numpy as np
from workstation.simulation.robot_driver import RobotMotion
from workstation.planning.collision import AABB
from workstation.planning.selection import failure_category, UPRIGHT_PLACE_TCP_CLEARANCE_M


class PathPreflightTests(unittest.TestCase):
    def driver(self):
        robot=RobotMotion.__new__(RobotMotion)
        states={name:SimpleNamespace(joint_positions_rad=np.zeros(7),finger_positions_m=np.full(2,.04)) for name in ('left','right')}
        robot.read_states=lambda:states
        robot.targets={name:np.r_[np.zeros(7),[.018,.018]] for name in states}
        robot.home=np.zeros(9);robot.place_support_z=.76;robot.place_tcp_clearance_m=UPRIGHT_PLACE_TCP_CLEARANCE_M;robot.static=[]
        robot.limits=(np.full(7,-10.),np.full(7,10.))
        robot.last_plan_check={};robot.spheres=lambda *args:[]
        return robot

    def test_entire_task_checked_before_motion(self):
        robot=self.driver(); robot.solve=lambda *args:np.zeros(7)
        robot.cartesian_knots=lambda *args:[np.zeros(7),np.ones(7)*.01]
        checks=[]
        robot.validate_path=lambda *args,**kw:checks.append((args,kw)) or .05
        obj=SimpleNamespace(size_m=(.1,.055,.045),uncertainty_m=.003)
        candidate=SimpleNamespace(arm='left',position_m=(0.,0.,.7825),yaw_rad=.3,target_id='observed')
        result=robot.feasibility(candidate,obj,None,(.2,-.4))
        self.assertTrue(result['whole_task_template_checked'])
        self.assertEqual(result['checked_phases'],['PREGRASP','APPROACH','LIFT','TRANSFER','LOWER','RETREAT','HOME'])
        self.assertEqual(len(checks),7)
        self.assertEqual(result['pending_observation_phases'],['LIFT','TRANSFER','LOWER','RETREAT'])
        self.assertTrue(all(kw.get('check_local_unknown') is False for _,kw in checks[2:6]))
        np.testing.assert_array_equal(checks[3][1]['finger_positions'],np.full(2,.0305))
        forecast=checks[5][1]['target_boxes_override'][0]
        self.assertEqual(forecast.name,'observed')
        np.testing.assert_allclose(forecast.centre,(.2,-.4,.7825))
        np.testing.assert_array_equal(robot.targets['left'][7:],[.018,.018])

    def test_hint_does_not_replace_measured_path_start_or_skip_phases(self):
        robot=self.driver();robot.solve=lambda arm,p,yaw,warm:warm.copy()
        robot.cartesian_knots=lambda *args:[np.ones(7)*.2,np.ones(7)*.21]
        checks=[]
        robot.validate_path=lambda arm,path,*args,**kw:checks.append(path) or .05
        candidate=SimpleNamespace(arm='left',position_m=(0.,0.,.7825),yaw_rad=0.,target_id='observed')
        result=robot.feasibility(candidate,SimpleNamespace(size_m=(.1,.055,.045),uncertainty_m=.003),None,
                                 (.2,-.4),pregrasp_hint=np.ones(7)*.2)
        self.assertEqual(len(result['checked_phases']),7)
        np.testing.assert_array_equal(checks[0][0],np.zeros(7))
        self.assertTrue(result['pregrasp_hint_used'])

    def test_held_carton_uses_actual_swept_rotation(self):
        robot=self.driver();a=np.pi/4;c,s=np.cos(a),np.sin(a)
        robot.fk=lambda *args:(np.array([0.,0.,1.]),np.array([[c,-s,0],[s,c,0],[0,0,-1.]]))
        robot.static=[AABB('corner',(.04,.04,1.),(.004,.004,.004))]
        scene=SimpleNamespace(objects=[])
        with self.assertRaisesRegex(RuntimeError,'HELD_CARTON_PATH_BLOCKED'):
            robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],scene,'observed',(.1,.03,.02),0.)

    def test_held_box_forearm_contact_not_exempted(self):
        robot=self.driver();robot.fk=lambda *args:(np.array([0.,0.,1.]),np.eye(3))
        robot.spheres=lambda arm,*args:[(6,np.array([.02,0,1.]),.01)] if arm == 'left' else []
        with self.assertRaisesRegex(RuntimeError,'HELD_CARTON_OWN_BODY_COLLISION'):
            robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'v',(.1,.055,.045))

    def test_failure_reasons_remain_distinct(self):
        self.assertEqual(failure_category('ROBOT_UNREACHABLE'),'ROBOT_REACHABILITY')
        self.assertEqual(failure_category('STATIC_PATH_BLOCKED:table'),'PATH_OR_COLLISION')
        self.assertEqual(failure_category('GRIPPER_SPACE_INSUFFICIENT'),'GRIPPER_OR_NEIGHBOUR_SPACE')
        self.assertEqual(failure_category('SIDE_POSTURE'),'POSTURE_TEMPLATE_UNAVAILABLE')

    def test_rotated_held_box_does_not_use_world_aabb_for_spheres(self):
        robot=self.driver();a=np.pi/4;c,s=np.cos(a),np.sin(a)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        robot.fk=lambda *args:(np.array([0.,0.,1.]),rotation@np.diag([1.,-1.,-1.]))
        # Inside the enclosing world AABB, outside the actual narrow box.
        robot.spheres=lambda arm,*args:[(8,np.array([.035,-.035,.998]),.004)] if arm == 'left' else []
        robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'v',(.1,.03,.02))

    def test_runtime_held_sweep_uses_latched_visual_relation(self):
        robot=self.driver();robot.fk=lambda *args:(np.array([0.,0.,1.]),np.eye(3))
        queries=[]
        robot.observed_workspace=SimpleNamespace(check_oriented_box=lambda p,s,r,**kw:queries.append(
            (p.copy(),s.copy(),r.copy(),kw)))
        robot.held_visual_geometry=dict(arm='left',target_id='observed',offset=np.array([.004,-.003,-.010]),
            relative_rotation=np.eye(3))
        robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'observed',(.1,.055,.045))
        np.testing.assert_allclose(queries[0][0],[.004,-.003,.990])
        np.testing.assert_allclose(queries[0][1],[.106,.061,.051])
        np.testing.assert_array_equal(queries[0][2],np.eye(3))
        self.assertTrue(queries[0][3]['check_unknown'])

    def test_nominal_palm_gap_is_preserved_without_waiving_actual_collision(self):
        robot=self.driver();robot.fk=lambda *args:(np.array([0.,0.,1.]),np.diag([1.,-1.,-1.]))
        robot.spheres=lambda arm,*args:[(8,np.array([0.,0.,1.05]),.024)] if arm=='left' else []
        path=[np.zeros(7),np.ones(7)*.01];scene=SimpleNamespace(objects=[])
        robot.validate_path('left',path,scene,'observed',(.1,.055,.045))
        # Fresh relation from trial 40: physical nominal gap is positive,
        # but insufficient after 3 mm uncertainty and 2 mm clearance.
        robot.held_visual_geometry=dict(arm='left',target_id='observed',offset=np.array([0.,0.,.00114]),
            relative_rotation=np.diag([1.,-1.,-1.]))
        with self.assertRaisesRegex(RuntimeError,'HELD_CARTON_OWN_BODY_COLLISION'):
            robot.validate_path('left',path,scene,'observed',(.1,.055,.045))
        self.assertGreater(robot.last_plan_check['nominal_clearance_m'],.002)
        self.assertLess(robot.last_plan_check['clearance_m'],.002)

    def test_held_sweep_retains_visual_uncertainty_and_rejects_unbounded_value(self):
        robot=self.driver();robot.fk=lambda *args:(np.array([0.,0.,1.]),np.eye(3))
        queries=[]
        robot.observed_workspace=SimpleNamespace(check_oriented_box=lambda p,s,r,**kw:queries.append(s.copy()))
        robot.held_visual_geometry=dict(arm='left',target_id='v',offset=np.zeros(3),
                                       relative_rotation=np.eye(3),uncertainty_m=.006)
        path=[np.zeros(7),np.ones(7)*.01];scene=SimpleNamespace(objects=[])
        robot.validate_path('left',path,scene,'v',(.1,.055,.045))
        np.testing.assert_allclose(queries[0],[.112,.067,.057])
        for value in (float('nan'),float('inf'),-.001,.009):
            robot.held_visual_geometry['uncertainty_m']=value
            with self.assertRaisesRegex(RuntimeError,'HELD_POSE_UNCERTAINTY_UNBOUNDED'):
                robot.validate_path('left',path,scene,'v',(.1,.055,.045))

    def test_payload_confirmation_uses_image_time_fk(self):
        robot=self.driver();queries=[];confirmed=[]
        packet=SimpleNamespace(robot_state={'left':SimpleNamespace(joint_positions_rad=np.full(7,.02))})
        robot.fk=lambda arm,q:(queries.append(q.copy()) or np.array([0.,0.,1.]),np.eye(3))
        robot.held_visual_geometry=dict(arm='left',target_id='visual_held')
        robot.observed_workspace=SimpleNamespace(packet=packet,confirm_payload=lambda *args:confirmed.append(args) or True)
        tracker=SimpleNamespace(offset=np.array([0.,0.,-.004]),relative_rotation=np.eye(3),
                                size=np.array([.1,.055,.045]),uncertainty_m=.006)
        self.assertTrue(robot.confirm_payload_observation('visual_held',tracker,{},packet))
        np.testing.assert_array_equal(queries[0],packet.robot_state['left'].joint_positions_rad)
        np.testing.assert_array_equal(confirmed[0][1],[0.,0.,.996])
        self.assertEqual(confirmed[0][-1],.006)
        self.assertFalse(robot.confirm_payload_observation('visual_held',tracker,{},SimpleNamespace()))
