import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from workstation.skills.visual_pick_place import VisualPickPlace


class ObserverScheduleTests(unittest.TestCase):
    def skill(self):
        skill=VisualPickPlace.__new__(VisualPickPlace)
        skill._monitor_ticks=0;skill.grasp_source=object();skill.held=False
        skill.priors=SimpleNamespace(max_age_s=.15);skill.robot=SimpleNamespace(dt=1/120)
        return skill

    def test_one_minute_uses_600_periodic_observations_instead_of_800(self):
        skill=self.skill();skill.observe=Mock()
        for _ in range(7200):skill.monitor()
        self.assertEqual(skill.observe.call_count,600)
        self.assertLessEqual(skill._monitor_period_steps()*skill.robot.dt+4*skill.robot.dt,
                             skill.priors.max_age_s)

    def test_successful_explicit_observation_restarts_period(self):
        skill=self.skill();packet=object();scene=object()
        skill.read_packet=lambda:packet;skill.estimator=SimpleNamespace(estimate=lambda _:scene)
        skill.robot.set_observation=Mock();skill._read_holding_surface=lambda:None
        skill._register_holding_surface=lambda value:value;skill.hold_tracker=None
        skill._anchor_normal_stage=Mock();skill.trace_scene=Mock();skill.save_scene=Mock()
        skill._monitor_ticks=11
        self.assertIs(skill.observe('pregrasp'),scene)
        self.assertEqual(skill._monitor_ticks,0)
        skill.observe=Mock()
        skill.monitor();skill.observe.assert_not_called()

    def test_reduced_age_budget_and_other_skills_keep_their_bounds(self):
        skill=self.skill()
        for dt in (1/120,1/240):
            skill.robot.dt=dt
            for age in (.09,.15,.3):
                skill.priors.max_age_s=age
                self.assertLessEqual(skill._monitor_period_steps()*dt,min(.1,age*2/3)+1e-9)
        skill.robot.dt=1/120;skill.priors.max_age_s=.15
        skill.accepted_postures=('SIDE',)
        self.assertEqual(skill._monitor_period_steps(),9)
        skill.grasp_source=None
        self.assertEqual(skill._monitor_period_steps(),24)
