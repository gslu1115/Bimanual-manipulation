"""Display keepalive must not change path checks or control stepping."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from tests import test_arm_observation_checks as fixtures
from workstation.simulation.display_refresh import DisplayRefresh


class DisplayRefreshTests(unittest.TestCase):
    def test_wall_clock_throttle_and_no_catch_up_burst(self):
        clock=[0.];render=Mock();refresh=DisplayRefresh(render,lambda:clock[0])
        self.assertFalse(refresh());clock[0]=.25;self.assertTrue(refresh())
        self.assertFalse(refresh());clock[0]=8.;self.assertTrue(refresh())
        self.assertFalse(refresh());self.assertEqual(render.call_count,2)

    def test_nested_render_is_not_reentered_and_errors_propagate(self):
        clock=[0.];nested=[]
        def render():nested.append(refresh());raise ValueError('renderer failed')
        refresh=DisplayRefresh(render,lambda:clock[0]);clock[0]=1.
        with self.assertRaisesRegex(ValueError,'renderer failed'):refresh()
        self.assertEqual(nested,[False]);self.assertFalse(refresh.active)

    def test_path_validation_keeps_checks_identical_without_driving(self):
        fixture=fixtures.ArmObservationChecksTests()
        a=fixture.driver(fixture.workspace());b=fixture.driver(fixture.workspace())
        clock=[0.];renders=[]
        b.refresh_display=DisplayRefresh(lambda:renders.append(clock[0]),lambda:clock[0])
        original=b.spheres
        def slow_geometry(*args):clock[0]+=.3;return original(*args)
        b.spheres=slow_geometry;b.step=Mock();b.command=Mock();b.monitor=Mock()
        path=[np.zeros(7),np.ones(7)*.2];scene=SimpleNamespace(objects=[])
        result_a=a.validate_path('left',path,scene,'v',target_contact=True)
        result_b=b.validate_path('left',path,scene,'v',target_contact=True)
        self.assertEqual(result_a,result_b);self.assertEqual(a.last_plan_check,b.last_plan_check)
        self.assertEqual(len(a.observed_workspace.checked),len(b.observed_workspace.checked))
        self.assertEqual(len(a.observed_workspace.unknown),len(b.observed_workspace.unknown))
        self.assertGreater(len(renders),2)
        b.step.assert_not_called();b.command.assert_not_called();b.monitor.assert_not_called()

    def test_control_tick_retains_exactly_one_step_and_one_observation(self):
        fixture=fixtures.ArmObservationChecksTests();robot=fixture.driver(fixture.workspace())
        robot.targets={'left':np.zeros(9)};robot.command=Mock();robot.step=Mock();robot.monitor=Mock()
        robot.refresh_display=Mock();robot.tick()
        robot.step.assert_called_once();robot.monitor.assert_called_once()
        robot.command.assert_called_once();robot.refresh_display.assert_called_once()


if __name__=='__main__':unittest.main()
