import unittest
from types import SimpleNamespace
import numpy as np
from tests import test_arm_observation_checks as fixtures


class PathSampleReuseTests(unittest.TestCase):
    def test_repeated_endpoint_reuses_only_identical_joint_sample(self):
        fixture=fixtures.ArmObservationChecksTests();ws=fixture.workspace();robot=fixture.driver(ws)
        path=[np.full(7,x) for x in (0.,.015625,.03125)]
        robot.validate_path('left',path,SimpleNamespace(objects=[]),'v',target_contact=True)
        self.assertEqual(robot.last_plan_check['samples'],4)
        self.assertEqual(robot.last_plan_check['unique_samples'],3)
        self.assertEqual(len(ws.checked),6)
        self.assertEqual(len(ws.unknown),3)

    def test_tiny_joint_difference_is_not_rounded_away(self):
        fixture=fixtures.ArmObservationChecksTests();robot=fixture.driver(fixture.workspace())
        path=[np.zeros(7),np.full(7,1e-12)]
        robot.validate_path('left',path,SimpleNamespace(objects=[]),'v')
        self.assertEqual(robot.last_plan_check['unique_samples'],2)

    def test_next_validation_uses_new_obstacle_evidence_even_at_same_joints(self):
        fixture=fixtures.ArmObservationChecksTests();ws=fixture.workspace();robot=fixture.driver(ws)
        path=[np.zeros(7),np.zeros(7)];scene=SimpleNamespace(objects=[])
        robot.validate_path('left',path,scene,'v')
        ws.check_observed_sphere_obstacles=lambda *args:'OBSERVED_PATH_OBSTACLE'
        with self.assertRaisesRegex(RuntimeError,'OBSERVED_PATH_OBSTACLE'):
            robot.validate_path('left',path,scene,'v')
