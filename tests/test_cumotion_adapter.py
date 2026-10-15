import contextlib
import io
import unittest
from types import SimpleNamespace
import numpy as np

from workstation.simulation.cumotion_driver import merge_named_command,CuMotionPickPlaceDriver
from workstation.skills.cumotion_pick_place import placement_goal
from workstation.app import build_parser,load_config


class CuMotionAdapterTests(unittest.TestCase):
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

if __name__=='__main__':unittest.main()
