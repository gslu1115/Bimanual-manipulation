import contextlib
import io
import unittest
from types import SimpleNamespace
import numpy as np

from workstation.simulation.cumotion_driver import merge_named_command,CuMotionPickPlaceDriver,model_pregrasp_offset,phase_tolerances
from workstation.skills.cumotion_pick_place import placement_goal
from workstation.app import build_parser,load_config


class CuMotionAdapterTests(unittest.TestCase):
    def test_dev15_error_fits_air_waypoint_but_contact_phases_remain_precise(self):
        for phase in ('LIFT','APPROACH_PLACE','RETREAT'):
            position,angle=phase_tolerances(phase)
            self.assertLess(.0033,position);self.assertLess(.0117,angle)
            self.assertLessEqual(position,.01);self.assertLessEqual(angle,np.deg2rad(2.))
        for phase in ('DESCEND_PLACE','RELEASE'):
            position,angle=phase_tolerances(phase)
            self.assertLess(.0031,position);self.assertLess(.0106,angle)
            self.assertEqual((position,angle),(.005,np.deg2rad(1.)))
        for phase in ('APPROACH_PICK','DESCEND_PICK','GRASP','UNKNOWN'):
            self.assertEqual(phase_tolerances(phase),(.003,.01))

    def test_phase_switch_restores_contact_gate_before_measured_state_evaluation(self):
        driver=object.__new__(CuMotionPickPlaceDriver);driver.time_s=0.;driver.controller_wall_s=0.;driver.calls=0
        driver.contact_hold_q=None;driver.arm='left';driver.robot=SimpleNamespace(targets={'left':np.zeros(9)})
        measured=object();driver.read_state=lambda:measured;seen=[]
        controller=SimpleNamespace(phase=SimpleNamespace(name='APPROACH_PLACE'),failed=False)
        def forward(state,goal,time_s):
            seen.append((state,controller._position_tolerance,controller._orientation_tolerance))
            # SDK advances after checking the current phase, not before.
            controller.phase.name='DESCEND_PLACE'
            return SimpleNamespace(joints=None)
        controller.forward=forward;driver.controller=controller
        driver.propose(.02);driver.propose(.02)
        self.assertEqual(seen,[(measured,*phase_tolerances('APPROACH_PLACE')),
                               (measured,*phase_tolerances('DESCEND_PLACE'))])

    def test_tilted_model_approach_is_not_replaced_by_vertical(self):
        d=np.array([.0792760849,.047702685,-.9957106113]);d/=np.linalg.norm(d)
        pick=np.array([.1,.2,.8]);distance=.18
        np.testing.assert_allclose(pick+[0,0,distance]+model_pregrasp_offset(d,distance),pick-d*distance)
        with self.assertRaises(ValueError):model_pregrasp_offset([0,0,-2],distance)

    def test_resync_uses_measured_state_and_elapsed_time(self):
        driver=object.__new__(CuMotionPickPlaceDriver);driver.time_s=2.
        measured=object();goal=object();seen=[]
        driver.read_state=lambda:measured
        driver.controller=SimpleNamespace(_make_arm_setpoint=lambda:goal,phase=SimpleNamespace(name='LIFT'))
        driver.arm_controller=SimpleNamespace(reset=lambda *args:seen.append(args) or True)
        driver.resync_after_cartesian(3.25)
        self.assertEqual(seen,[(measured,goal,5.25)])
        self.assertIsNone(driver.contact_hold_q)
        with self.assertRaises(ValueError):driver.resync_after_cartesian(-1.)

    def test_grasp_holds_reached_arm_without_hiding_measured_pose_or_overriding_fingers(self):
        driver=object.__new__(CuMotionPickPlaceDriver);driver.time_s=2.;driver.controller_wall_s=0.;driver.calls=0
        driver.arm='left';driver.names=['j'+str(i) for i in range(9)]
        q=np.arange(7)*.01;target=np.r_[q,[.04,.04]]
        driver.robot=SimpleNamespace(targets={'left':target},
            read_states=lambda:{'left':SimpleNamespace(joint_positions_rad=q)})
        measured=object();seen=[];driver.read_state=lambda:measured
        proposed=np.r_[q+.1,[.023,.023]];vel=np.ones(9)*.2
        joints=SimpleNamespace(position_names=driver.names,velocity_names=driver.names,
            positions=SimpleNamespace(numpy=lambda:proposed),velocities=SimpleNamespace(numpy=lambda:vel))
        controller=SimpleNamespace(phase=SimpleNamespace(name='DESCEND_PICK'),failed=False,
            _make_arm_setpoint=lambda:None,forward=lambda state,*args:seen.append(state) or SimpleNamespace(joints=joints))
        driver.controller=controller;driver.arm_controller=SimpleNamespace(reset=lambda *a:True)
        driver.resync_after_cartesian(3.)
        for phase in ('DESCEND_PICK','GRASP'):
            controller.phase.name=phase
            command,velocity=driver.propose(.02)
            np.testing.assert_allclose(command[:7],q);np.testing.assert_allclose(velocity[:7],0.)
            np.testing.assert_allclose(command[7:],proposed[7:]);np.testing.assert_allclose(velocity[7:],vel[7:])
        self.assertEqual(seen,[measured,measured])
        controller.phase.name='LIFT';driver.resync_after_cartesian(2.)
        command,velocity=driver.propose(.02)
        np.testing.assert_allclose(command,proposed);np.testing.assert_allclose(velocity,vel)

    def test_named_output_is_copied_and_uncommanded_fingers_preserved(self):
        current=np.array([1.,2.,.03,.03]);values=np.array([8.,7.])
        result=merge_named_command(current,['b','a'],values,['a','b','f1','f2'])
        values[:]=99
        np.testing.assert_allclose(result,[7.,8.,.03,.03])
        np.testing.assert_allclose(current,[1.,2.,.03,.03])

    def test_invalid_output_is_rejected_before_command(self):
        for names,values in [(['bad'],[0.]),(['a','a'],[0.,0.]),(['a'],[float('nan')]),(['a'],[])]:
            with self.subTest(names=names,values=values),self.assertRaises(ValueError):
                merge_named_command([0.],names,values,['a'])

    def test_visual_grasp_relation_places_body_at_slot(self):
        a=.73;c,s=np.cos(a),np.sin(a)
        body=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        tcp_r=np.diag([1.,-1.,-1.])
        relative=tcp_r.T@body;offset=np.array([.007,-.003,.005])
        p,r=placement_goal((.2,.3),.8,(.1,.06,.045),.015,body,offset,relative)
        np.testing.assert_allclose(p+r@offset,[.2,.3,.8375],atol=1e-10)
        np.testing.assert_allclose(r@relative,np.eye(3),atol=1e-10)

    def test_goal_update_cannot_rewind_grasp_or_modify_release(self):
        driver=object.__new__(CuMotionPickPlaceDriver)
        for phase in ('APPROACH_PICK','GRASP','RELEASE','RETREAT','DONE'):
            driver.controller=SimpleNamespace(phase=SimpleNamespace(name=phase))
            with self.subTest(phase=phase),self.assertRaisesRegex(RuntimeError,'OUTSIDE_OBSERVED_BOUNDARY'):
                driver.update_place_from_measurement([0,0,0],np.eye(3))

    def test_backend_does_not_change_default(self):
        parser=build_parser();args=parser.parse_args(['--validate-only'])
        self.assertEqual(load_config(args,parser)['motion_backend'],'legacy')

    def test_backend_requires_controlled_model_skill(self):
        parser=build_parser()
        for options in (['--motion-backend','cumotion'],
                        ['--mode','visual-pnp','--motion-backend','cumotion']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                load_config(parser.parse_args(options),parser)

    def test_backend_keeps_camera_installation(self):
        parser=build_parser()
        common=['--mode','visual-pnp','--grasp-backend','graspgen','--graspgen-assets','assets']
        legacy=load_config(parser.parse_args(common),parser)
        official=load_config(parser.parse_args(common+['--motion-backend','cumotion']),parser)
        self.assertEqual(legacy['cameras'],official['cameras'])
        self.assertEqual(legacy['robots'],official['robots'])

    def test_manual_start_requires_visible_official_trial(self):
        parser=build_parser()
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            load_config(parser.parse_args(['--wait-for-start']),parser)
        args=parser.parse_args(['--mode','visual-pnp','--grasp-backend','graspgen',
            '--graspgen-assets','assets','--motion-backend','cumotion','--livestream','--wait-for-start'])
        self.assertTrue(load_config(args,parser)['wait_for_start'])

if __name__=='__main__':unittest.main()
