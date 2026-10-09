import unittest
from types import SimpleNamespace
import numpy as np
from workstation.observations.observation_packet import RobotState
from workstation.simulation.robot_driver import RobotMotion


class ArmObservationChecksTests(unittest.TestCase):
    def driver(self,workspace):
        robot=RobotMotion.__new__(RobotMotion)
        state=RobotState(1.,np.zeros(7),np.full(2,.04))
        robot.read_states=lambda:{'left':state,'right':state}
        robot.limits=(np.full(7,-3.),np.full(7,3.));robot.static=[];robot.last_plan_check={}
        robot.spheres=lambda arm,*args:[(5,np.array([0,0,1.]),.03),(9,np.array([0,.2,1.]),.009)] if arm=='left' else []
        robot.observed_workspace=workspace
        robot.fk=lambda *args:(np.array([0.,0.,1.]),np.eye(3))
        return robot

    def workspace(self):
        checked=[];unknown=[]
        return SimpleNamespace(checked=checked,unknown=unknown,
            check_observed_sphere_obstacles=lambda p,r:checked.append((p,r)),
            check_sphere=lambda p,r:unknown.append((p,r)),
            last_unknown_sample=None,report=lambda:{'scope':'fixture'})

    def test_body_obstacles_and_tcp_corridor_remain(self):
        workspace=self.workspace();robot=self.driver(workspace)
        robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'visual_1',target_contact=True)
        self.assertEqual(len(workspace.checked),4)
        self.assertEqual(len(workspace.unknown),2)
        self.assertFalse(robot.last_plan_check['whole_arm_unknown_space_verified'])

    def test_empty_transfer_checks_obstacles_without_full_visibility_gate(self):
        workspace=self.workspace();robot=self.driver(workspace)
        robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'visual_1')
        self.assertEqual(len(workspace.checked),4)
        self.assertFalse(workspace.unknown)
        self.assertFalse(robot.last_plan_check['approach_corridor_unknown_space_checked'])

    def test_visibility_query_is_tcp_corridor_not_inflated_finger_shell(self):
        workspace=self.workspace();robot=self.driver(workspace)
        robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'visual_1',target_contact=True)
        for p,r in workspace.unknown:
            np.testing.assert_array_equal(p,[0.,0.,1.]);self.assertEqual(r,.003)
        self.assertFalse(robot.last_plan_check['finger_shell_visibility_required'])

    def test_observed_forearm_obstacle_still_blocks(self):
        workspace=self.workspace();workspace.check_observed_sphere_obstacles=lambda *args:'OBSERVED_PATH_OBSTACLE'
        robot=self.driver(workspace)
        with self.assertRaisesRegex(RuntimeError,'OBSERVED_PATH_OBSTACLE'):
            robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'visual_1')

    def test_never_observed_tcp_corridor_still_blocks(self):
        workspace=self.workspace();workspace.check_sphere=lambda *args:'UNOBSERVED_PATH_SPACE'
        robot=self.driver(workspace)
        with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):
            robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'visual_1',target_contact=True)

    def test_contact_permission_does_not_make_empty_retreat_an_approach(self):
        workspace=self.workspace();workspace.check_sphere=lambda *args:self.fail('approach gate applied to retreat')
        robot=self.driver(workspace)
        path=[np.zeros(7),np.ones(7)*.01];scene=SimpleNamespace(objects=[])
        robot.validate_path('left',path,scene,'v',target_contact=True,check_approach_corridor=False)
        self.assertEqual(len(workspace.checked),4)
        self.assertFalse(robot.last_plan_check['approach_corridor_unknown_space_checked'])
        self.assertFalse(robot.last_plan_check['empty_motion_unknown_space_verified'])
        workspace.check_observed_sphere_obstacles=lambda *args:'OBSERVED_PATH_OBSTACLE'
        with self.assertRaisesRegex(RuntimeError,'OBSERVED_PATH_OBSTACLE'):
            robot.validate_path('left',path,scene,'v',target_contact=True,check_approach_corridor=False)

    def test_forecast_visibility_deferral_is_explicit_and_not_runtime_default(self):
        workspace=self.workspace();workspace.check_sphere=lambda *args:'UNOBSERVED_PATH_SPACE'
        robot=self.driver(workspace)
        robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'visual_1',
                            target_contact=True,check_local_unknown=False)
        self.assertTrue(robot.last_plan_check['local_unknown_space_check_deferred'])
        with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):
            robot.validate_path('left',[np.zeros(7),np.ones(7)*.01],SimpleNamespace(objects=[]),'visual_1',target_contact=True)


if __name__=='__main__':unittest.main()
