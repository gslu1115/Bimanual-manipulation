import unittest
from types import SimpleNamespace
import numpy as np
from workstation.observations.observation_packet import RobotState
from workstation.skills.reobserve import GuidedReobserve


class GuidedReobserveTests(unittest.TestCase):
    def setup_case(self,finger=.04):
        state=RobotState(1.,np.zeros(7),np.full(2,finger))
        robot=SimpleNamespace(read_states=lambda:{'left':state},hold=lambda _:None,
            solve=lambda *args:np.ones(7)*.2,
            observation_prefix=lambda *args:(np.ones(7)*.1,{'checked':True}))
        moves=[];robot.execute_joint=lambda *args:moves.append(args)
        scene=SimpleNamespace(observation_time_s=1.,objects=[SimpleNamespace(track_id='visual_1',failures=())])
        candidate=SimpleNamespace(target_id='visual_1',arm='left',position_m=(0,0,.8),yaw_rad=0)
        return robot,scene,candidate,moves

    def test_only_checked_prefix_moves_before_fresh_observation(self):
        robot,scene,candidate,moves=self.setup_case();observations=[]
        def observe(label=None):observations.append(label);return scene
        _,report=GuidedReobserve(robot,observe).run(lambda scene:True,scene,lambda:[candidate])
        self.assertTrue(report['success']);self.assertEqual(len(moves),1)
        self.assertEqual(len(observations),2)
        np.testing.assert_allclose(moves[0][1][:7],np.full(7,.1))

    def test_rejected_prefix_never_moves(self):
        robot,scene,candidate,moves=self.setup_case()
        def reject(*args):raise RuntimeError('NO_OBSERVED_VIEWPOINT_PREFIX')
        robot.observation_prefix=reject
        _,report=GuidedReobserve(robot,lambda *args:self.fail('unexecuted observation')).run(
            lambda _:False,scene,lambda:[candidate])
        self.assertFalse(report['executed']);self.assertFalse(moves)
        self.assertEqual(report['iterations'],1)

    def test_closed_unknown_load_never_explores(self):
        robot,scene,candidate,moves=self.setup_case(.02)
        _,report=GuidedReobserve(robot,lambda *args:scene).run(lambda _:True,scene,lambda:[candidate])
        self.assertEqual(report['reason'],'GRIPPER_NOT_EMPTY_AND_OPEN');self.assertFalse(moves)
