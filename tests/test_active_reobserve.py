import unittest
from types import SimpleNamespace
import numpy as np
from workstation.skills.reobserve import ActiveReobserve


class ActiveReobserveTests(unittest.TestCase):
    def robot(self,fingers=.04):
        robot=SimpleNamespace(limits=(np.full(7,-3.),np.full(7,3.)))
        robot.read_states=lambda:{'left':SimpleNamespace(qpos=np.r_[np.zeros(7),[fingers,fingers]])}
        robot.hold=lambda seconds:None
        robot.execute_joint=lambda *args:None
        return robot

    def test_empty_wrist_moves_and_uses_new_observation(self):
        robot=self.robot();moves=[];robot.execute_joint=lambda *args:moves.append(args)
        times=[]
        def observe(label=None):
            times.append(label);return SimpleNamespace(observation_time_s=len(times))
        scene,result=ActiveReobserve(robot,observe).run(lambda scene:True,None)
        self.assertTrue(result['success']);self.assertTrue(result['active_camera_motion'])
        self.assertEqual(len(times),2);self.assertEqual(moves[0][1][6],.30)

    def test_uncertain_load_never_rotates(self):
        robot=self.robot(.02)
        robot.execute_joint=lambda *args:self.fail('uncertain load moved')
        _,result=ActiveReobserve(robot,lambda *args:None).run(lambda scene:True,None)
        self.assertFalse(result['executed']);self.assertEqual(result['reason'],'GRIPPER_NOT_EMPTY_AND_OPEN')

    def test_unknown_space_rejections_are_retained(self):
        robot=self.robot()
        def reject(*args):raise RuntimeError('UNOBSERVED_PATH_SPACE')
        robot.execute_joint=reject
        _,result=ActiveReobserve(robot,lambda *args:self.fail('rejected motion observed')).run(lambda scene:True,None)
        self.assertFalse(result['success']);self.assertFalse(result['active_camera_motion'])
        self.assertEqual(len(result['samples']),2)
        self.assertTrue(all(s['rejection']=='UNOBSERVED_PATH_SPACE' for s in result['samples']))

    def test_wrist_rotation_keeps_the_new_lifted_pose(self):
        robot=self.robot();q=np.r_[np.zeros(7),[.04,.04]];moves=[]
        robot.read_states=lambda:{'left':SimpleNamespace(qpos=q.copy())}
        def lift(*args):q[1]-=.3
        def move(arm,target,*args):moves.append(target.copy());q[:]=target
        robot.execute_observation_lift=lift;robot.execute_joint=move
        scene=SimpleNamespace(observation_time_s=1.)
        _,result=ActiveReobserve(robot,lambda *args:scene).run(lambda scene:bool(moves),scene)
        self.assertTrue(result['success']);self.assertAlmostEqual(moves[0][1],-.6)
        self.assertAlmostEqual(moves[0][6],.3)
