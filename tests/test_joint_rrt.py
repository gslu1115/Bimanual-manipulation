import unittest
import numpy as np
from workstation.planning.joint_rrt import bounded_joint_rrt


class JointRRTTests(unittest.TestCase):
    def test_routes_around_blocked_straight_edge(self):
        def valid(a,b):
            points=np.linspace(a,b,80)
            return bool(np.all(np.linalg.norm(points,axis=1)>.18))
        path,report=bounded_joint_rrt([-.5,0],[.5,0],([-1,-1],[1,1]),valid,seed=3,iterations=600,seconds=2.)
        self.assertIsNotNone(path);self.assertIsNone(report['reason'])
        np.testing.assert_allclose(path[0],[-.5,0]);np.testing.assert_allclose(path[-1],[.5,0])
        self.assertTrue(all(valid(a,b) for a,b in zip(path,path[1:])))

    def test_unknown_space_is_never_relaxed(self):
        path,report=bounded_joint_rrt([-.5,0],[.5,0],([-1,-1],[1,1]),lambda a,b:False)
        self.assertIsNone(path);self.assertEqual(report['reason'],'GOAL_STATE_NOT_VERIFIED')

    def test_bounded_search_does_not_claim_unreachable(self):
        def valid(a,b):return bool(np.all(a==b))
        path,report=bounded_joint_rrt([0,0],[.5,0],([-1,-1],[1,1]),valid,iterations=3)
        self.assertIsNone(path);self.assertFalse(report['unreachable_proven'])

    def test_joint_limits_are_enforced(self):
        path,report=bounded_joint_rrt([0,0],[2,0],([-1,-1],[1,1]),lambda a,b:True)
        self.assertIsNone(path);self.assertEqual(report['reason'],'JOINT_LIMIT')

    def test_nonfinite_joint_input_never_reaches_collision_callback(self):
        path,report=bounded_joint_rrt([0,0],[float('nan'),0],([-1,-1],[1,1]),
                                    lambda *args:self.fail('invalid input accepted'))
        self.assertIsNone(path);self.assertEqual(report['reason'],'INVALID_JOINT_INPUT')

    def test_start_state_must_also_be_verified(self):
        path,report=bounded_joint_rrt([0,0],[.5,0],([-1,-1],[1,1]),lambda a,b:bool(a[0]>.1))
        self.assertIsNone(path);self.assertEqual(report['reason'],'START_STATE_NOT_VERIFIED')
