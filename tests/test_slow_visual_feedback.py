import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from workstation.app import build_parser,load_config
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.skills.visual_pick_place import VisualPickPlace
from tests.test_scene_estimate import packet
from tests import test_robot_driver


class SlowVisualFeedbackTests(unittest.TestCase):
    def skill(self,period=None):
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        return VisualPickPlace(SimpleNamespace(dt=1/120),lambda:None,priors,[],
            grasp_source=object(),observation_period_s=period)

    def test_default_one_second_and_explicit_two_second_schedule(self):
        self.assertEqual(self.skill().observation_period_s,1.)
        self.assertEqual(self.skill(.1).robot.visual_feedback_max_age_s,.15)
        for period,count in ((.1,600),(1.,60),(2.,30)):
            skill=self.skill(period);skill.observe=Mock()
            for _ in range(7200):skill.monitor()
            self.assertEqual(skill.observe.call_count,count)
            self.assertEqual(skill.priors.max_age_s,.15)

    def test_motion_may_use_old_support_until_deadline_without_rewriting_time(self):
        for period in (1.,2.):
            skill=self.skill(period)
            driver,_,_,_,clock=test_robot_driver.FullPoseDriveTests.driver()
            driver.visual_feedback_max_age_s=skill.robot.visual_feedback_max_age_s
            clock[0]=1.+period
            driver._require_rotating_payload('left','v',(.1,.055,.045))
            self.assertEqual(driver.observed_workspace.payload_observation['time_s'],1.)
            self.assertEqual(driver.observed_workspace.packet.observation_time_s,1.)
            clock[0]=1.+driver.visual_feedback_max_age_s+.001
            with self.assertRaisesRegex(RuntimeError,'GEOMETRY_STALE'):
                driver._require_rotating_payload('left','v',(.1,.055,.045))

    def test_late_received_image_and_revoked_geometry_still_reject(self):
        skill=self.skill(2.)
        for revoke in (False,True):
            driver,_,_,_,clock=test_robot_driver.FullPoseDriveTests.driver()
            driver.visual_feedback_max_age_s=skill.robot.visual_feedback_max_age_s
            clock[0]=2.
            if revoke:driver.observed_workspace.payload_observation=None
            else:driver.observed_workspace.packet.assembled_time_s=1.2
            with self.assertRaisesRegex(RuntimeError,'GEOMETRY_UNSUPPORTED'):
                driver._require_rotating_payload('left','v',(.1,.055,.045))

    def test_stationary_identity_survives_interval_but_stale_images_do_not(self):
        for period in (1.,2.):
            skill=self.skill(period);est=skill.estimator
            old=est.estimate(packet(time=1.)).objects[0]
            current=est.estimate(packet(time=1.+period)).objects[0]
            self.assertEqual(old.track_id,current.track_id)
            self.assertEqual(current.time_s,1.+period)
            stale=est.estimate(packet(time=1.+period,assembled=1.+period+.2))
            self.assertFalse(stale.objects)
            self.assertEqual(stale.camera_status['scene_camera'],'STALE')

    def test_phase_capture_can_prove_short_lift_before_next_period(self):
        skill=self.skill(2.);p=object();scene=object()
        skill.read_packet=lambda:p;skill.estimator=SimpleNamespace(estimate=lambda _:scene)
        skill.robot.set_observation=Mock();skill.hold_tracker=object()
        skill._read_holding_surface=lambda:dict(height_gain_m=.12,grasp_constrained=False)
        skill._register_holding_surface=lambda value:value;skill._anchor_normal_stage=Mock()
        skill._monitor_ticks=100;skill.observe('lifted')
        self.assertEqual(skill.max_visual_lift,.12);self.assertEqual(skill._monitor_ticks,0)
        skill._read_holding_surface=lambda:dict(height_gain_m=.3,grasp_constrained=True)
        skill.observe('lifted');self.assertEqual(skill.max_visual_lift,.12)

    def test_cli_binds_interval_without_relaxing_received_frame_freshness(self):
        for period in (.1,1.,2.):
            parser=build_parser()
            args=parser.parse_args(['--mode','visual-pnp','--grasp-backend','graspgen',
                '--graspgen-assets','.','--visual-observation-period-s',str(period)])
            config=load_config(args,parser)
            self.assertEqual(config['visual_observation_period_s'],period)
            self.assertEqual(config['policy_observation']['max_frame_age_s'],.15)
        for period in ('0','3','nan'):
            parser=build_parser()
            args=parser.parse_args(['--mode','visual-pnp','--grasp-backend','graspgen',
                '--graspgen-assets','.','--visual-observation-period-s',period])
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                load_config(args,parser)
