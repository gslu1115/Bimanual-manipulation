import math
import unittest
from unittest.mock import Mock
import numpy as np
from tests import test_slow_visual_feedback


class AdaptiveObserverTests(unittest.TestCase):
    def skill(self,period=2.):
        return test_slow_visual_feedback.SlowVisualFeedbackTests().stable_transfer(period)

    def test_only_supported_lifted_translation_is_slow(self):
        skill=self.skill()
        self.assertEqual(skill._monitor_period_steps(),240)
        for phase in ('PREGRASP','APPROACH','CLOSE','PRELOAD','LIFT','LOWER','OPEN',
                      'RETREAT','HOME','VERIFY','FLIP','UNKNOWN'):
            skill.events[-1]['phase']=phase
            self.assertEqual(skill._monitor_period_steps(),12,phase)
        for attr,value in (('held',False),('releasing',True),('max_visual_lift',.07),
                           ('_missing',1),('_translation_only',False),('packet',object()),
                           ('_hold_observation',None)):
            skill=self.skill();setattr(skill,attr,value)
            self.assertEqual(skill._monitor_period_steps(),12,attr)
        skill=self.skill()
        skill._hold_observation=(*skill._hold_observation[:3],None)
        self.assertEqual(skill._monitor_period_steps(),12)
        # Permitted grasp-constrained support is not rejected just for occlusion.
        skill=self.skill();skill._hold_observation[3]['grasp_constrained']=True
        self.assertEqual(skill._monitor_period_steps(),240)

    def test_slow_to_fast_is_immediate_when_overdue_and_tightens_motion_deadline(self):
        skill=self.skill();skill._monitor_ticks=61
        skill._translation_only=False
        skill.monitor()
        skill.observe.assert_called_once()
        self.assertEqual(skill._monitor_ticks,0)
        self.assertEqual(skill.robot.visual_feedback_max_age_s,.15)

    def test_quaternion_rotation_overrides_transfer_phase_and_cleans_up_on_error(self):
        for degrees,sign,steps in ((0,1,240),(0,-1,240),(4,1,240),(6,1,12),(180,1,12)):
            skill=self.skill();skill._translation_only=False
            angle=math.radians(degrees)/2
            target=sign*np.array([math.cos(angle),math.sin(angle),0,0])
            actual=[]
            def execute(*args):
                actual.append(skill._monitor_period_steps())
                raise RuntimeError('test_motion_failure')
            skill._move=execute
            with self.assertRaisesRegex(RuntimeError,'test_motion_failure'):
                skill.move([0,0,0],0,0,orientation=target)
            self.assertEqual(actual,[steps],degrees)
            self.assertFalse(skill._translation_only)
            self.assertEqual(skill.robot.visual_feedback_max_age_s,.15)

    def test_reorienting_motion_refreshes_before_execution_when_overdue(self):
        skill=self.skill();skill._translation_only=False;skill._monitor_ticks=61
        order=[]
        skill.observe=Mock(side_effect=lambda:order.append('capture'))
        skill._move=lambda *args:order.append('motion')
        skill.move([0,0,0],0,math.pi/2)
        self.assertEqual(order,['capture','motion'])
        self.assertFalse(skill._translation_only)

    def test_mixed_schedule_counts_and_periodic_capture_loss_switch_to_fast(self):
        for period,expected in ((1.,330),(2.,315)):
            skill=self.skill(period)
            # 30 seconds critical + 30 seconds supported translation.
            skill._translation_only=False
            for _ in range(3600):skill.monitor()
            skill._translation_only=True
            for _ in range(3600):skill.monitor()
            self.assertEqual(skill.observe.call_count,expected)
            self.assertEqual(skill.periodic_observations,
                             dict(fast=300,stable_transfer=int(30/period)))
        skill=self.skill();skill._monitor_ticks=239
        def lost_support():
            skill._hold_observation=(*skill._hold_observation[:3],None)
            return skill.scene
        skill.observe=Mock(side_effect=lost_support)
        skill.initial_position=np.array([0,0,0])
        skill.monitor()
        self.assertEqual(skill._missing,1)
        self.assertEqual(skill._monitor_period_steps(),12)
        self.assertEqual(skill.robot.visual_feedback_max_age_s,.15)
