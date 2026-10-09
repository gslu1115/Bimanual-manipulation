import unittest
from types import SimpleNamespace
import numpy as np
from workstation.observations.observation_packet import RobotState
from workstation.skills.camera_reobserve import GapReobserve


class GapReobserveTests(unittest.TestCase):
    def fixture(self,finger=.04):
        robot=SimpleNamespace(read_states=lambda:{'panda_left':RobotState(1.,np.zeros(7),np.full(2,.04)),
            'panda_right':RobotState(1.,np.zeros(7),np.full(2,finger))},
            hold=lambda seconds:None,last_plan_check={})
        packet=SimpleNamespace(cameras={'right_wrist_camera':object()},camera_status={'right_wrist_camera':'OK'})
        robot.observed_workspace=SimpleNamespace(_frames=lambda:[packet.cameras['right_wrist_camera']],_ray_free=lambda *args:np.array([True]))
        scene=object();target=SimpleNamespace(position_m=(-.2,-.06,.7825),track_id='visual_1')
        priors=SimpleNamespace(support_z_m=.76)
        return robot,packet,scene,target,priors

    def test_only_other_open_arm_moves_and_point_is_reobserved(self):
        robot,packet,scene,target,priors=self.fixture();calls=[]
        robot.execute_camera_view=lambda *args:calls.append(args) or {'checked':True}
        observations=[]
        def observe(label=None):observations.append(label);return scene
        _,report=GapReobserve(robot,observe,lambda:packet,priors).run(
            'panda_left',[-.19,.04,1.116],target,scene)
        self.assertTrue(report['success']);self.assertEqual(calls[0][0],'panda_right')
        self.assertEqual(len(observations),2);self.assertIn('entire task',report['scope'])

    def test_closed_observer_never_moves(self):
        robot,packet,scene,target,priors=self.fixture(.02)
        robot.execute_camera_view=lambda *args:self.fail('unknown load moved')
        _,report=GapReobserve(robot,lambda *args:scene,lambda:packet,priors).run(
            'panda_left',[-.19,.04,1.116],target,scene)
        self.assertFalse(report['executed'])

    def test_all_unknown_path_rejections_retained(self):
        robot,packet,scene,target,priors=self.fixture()
        def reject(*args):raise RuntimeError('UNOBSERVED_PATH_SPACE')
        robot.execute_camera_view=reject
        _,report=GapReobserve(robot,lambda *args:self.fail('rejected movement observed'),lambda:packet,priors).run(
            'panda_left',[-.19,.04,1.116],target,scene)
        self.assertFalse(report['success']);self.assertEqual(len(report['samples']),3)
        self.assertTrue(all(s['reason']=='UNOBSERVED_PATH_SPACE' for s in report['samples']))

    def test_other_camera_cannot_stand_in_for_proposed_observer(self):
        robot,packet,scene,target,priors=self.fixture()
        robot.observed_workspace._frames=lambda:[object()]
        robot.execute_camera_view=lambda *args:{}
        _,report=GapReobserve(robot,lambda *args:scene,lambda:packet,priors).run(
            'panda_left',[-.19,.04,1.116],target,scene)
        self.assertFalse(report['success'])

    def test_preparation_centres_robot_point_and_stages_front_view(self):
        robot,packet,scene,target,priors=self.fixture();calls=[]
        robot.execute_camera_view=lambda *args:calls.append(args) or {}
        point=np.array([-.19,.04,1.116])
        _,report=GapReobserve(robot,lambda *args:scene,lambda:packet,priors).run(
            'panda_left',point,target,scene,prepared=True)
        self.assertTrue(report['success'])
        np.testing.assert_array_equal(calls[0][3],point)
        self.assertLess(calls[0][2][1],point[1])

    def test_preparation_keeps_bounded_complementary_view_proposals(self):
        robot,packet,scene,target,priors=self.fixture();calls=[]
        robot.execute_camera_view=lambda *args:calls.append(args) or {}
        robot.observed_workspace._ray_free=lambda *args:np.array([False])
        _,report=GapReobserve(robot,lambda *args:scene,lambda:packet,priors).run(
            'panda_left',[-.19,.04,1.116],target,scene,prepared=True)
        self.assertFalse(report['success']);self.assertEqual(len(calls),3)
        self.assertLess(calls[0][2][1],.04)
        self.assertGreater(calls[1][2][1],.12)
