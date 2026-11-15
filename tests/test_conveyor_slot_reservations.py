import unittest
from types import SimpleNamespace

from workstation.planning.task_targets import ConveyorSlots,reobserve_reserved_slot


class ConveyorSlotTests(unittest.TestCase):
    def reobserve(self,views,adjust=lambda candidate:True,candidates=(1,2)):
        goal=(.2,-.4);slots=ConveyorSlots({'conveyor':{'slots_xy':[goal]}})
        slots.owners[goal]='right';stream=iter(views)
        return reobserve_reserved_slot(slots,'right',goal,90.,lambda:next(stream),candidates,adjust,[])

    def view(self,time,free=False,**kwargs):
        return dict(xy=(.2,-.4),free=free,reason=None if free else 'UNOBSERVED_PATH_SPACE',
                    observation_time_s=time,observed_objects=[],**kwargs)

    def test_new_free_depth_needs_no_viewpoint_motion(self):
        self.assertEqual(self.reobserve([self.view(91.,True)],
            lambda c:self.fail('should not move'))['observation_time_s'],91.)

    def test_reobserve_skips_blocked_plans_then_uses_new_depth(self):
        moved=[]
        def adjust(c):moved.append(c);return c==2
        result=self.reobserve([self.view(91.),self.view(93.,True)],adjust)
        self.assertEqual(moved,[1,2]);self.assertEqual(result['observation_time_s'],93.)

    def test_viewpoint_execution_failure_never_falls_through_to_another_move(self):
        moved=[]
        def adjust(c):moved.append(c);raise RuntimeError('JOINT_TRACKING_ERROR')
        with self.assertRaisesRegex(RuntimeError,'JOINT_TRACKING_ERROR'):
            self.reobserve([self.view(91.)],adjust)
        self.assertEqual(moved,[1])

    def test_stale_wrong_or_occupied_views_cannot_refresh_evidence(self):
        for view in (self.view(90.,True),dict(self.view(91.,True),xy=(0.,-.4)),
                     dict(self.view(91.),observed_objects=['box'],reason='OBSERVED_SLOT_OCCUPIED')):
            with self.subTest(view=view),self.assertRaises(RuntimeError):
                self.reobserve([view],lambda c:self.fail('must stop without moving'))

    def test_no_visible_pose_is_bounded_and_fails_closed(self):
        moved=[]
        with self.assertRaisesRegex(RuntimeError,'NO_VISIBLE_CHECKED_POSE'):
            self.reobserve([self.view(91.),self.view(92.),self.view(93.)],lambda c:moved.append(c) or True)
        self.assertEqual(moved,[1,2])

    def test_reobservation_refreshes_timestamp_without_extending_twenty_second_limit(self):
        goal=(.2,-.4);slots=ConveyorSlots({'conveyor':{'slots_xy':[goal]}});slots.owners[goal]='right'
        fresh=self.reobserve([self.view(91.,True)])
        self.assertIsNotNone(slots.approach_clearance('right',goal,self.view(110.),fresh,accounted_for=True,stationary_belt=True))
        self.assertIsNone(slots.approach_clearance('right',goal,self.view(112.),fresh,accounted_for=True,stationary_belt=True))

    def test_occluded_reserved_slot_requires_recent_free_view_and_both_cartons(self):
        goal=(-.2,-.4)
        slots=ConveyorSlots({'conveyor':{'slots_xy':[goal,(.2,-.4)]}})
        slots.owners[goal]='left'
        prior=dict(xy=goal,free=True,observation_time_s=10.)
        view=dict(xy=goal,free=False,reason='UNOBSERVED_PATH_SPACE',observation_time_s=15.,observed_objects=[])
        def check(current=view,arm='left',accounted=True,stopped=True):
            return slots.approach_clearance(arm,goal,current,prior,accounted_for=accounted,stationary_belt=stopped)
        self.assertIsNotNone(check())
        for blocked in (dict(view,reason='OBSERVED_PATH_OBSTACLE'),dict(view,observed_objects=['other']),
                        dict(view,observation_time_s=31.)):
            self.assertIsNone(check(blocked))
        self.assertIsNone(check(accounted=False))
        self.assertIsNone(check(stopped=False))
        self.assertIsNone(check(arm='right'))

    def test_reserve_recheck_place_and_clear(self):
        positions = [(-.2, -.4), (0., -.4), (.2, -.4)]
        slots = ConveyorSlots({'conveyor': {'slots_xy': positions}})
        seen = {xy: [] for xy in positions}
        timestamp = 1.
        def observe():
            slots.observe(SimpleNamespace(slot=lambda xy: dict(
                free=seen[xy] == [], observed_objects=seen[xy] or [],
                observation_time_s=timestamp)))
        observe()
        left, middle, right = positions
        with self.assertRaises(ValueError):
            slots.reserve_pair({'left': middle, 'right': middle})
        slots.reserve_pair({'left': middle, 'right': right})
        self.assertEqual(slots.states[left], 'free')
        timestamp += 1
        seen[middle] = None
        observe()
        self.assertEqual(slots.owners[middle], 'left')
        self.assertEqual(slots.states[middle], 'unknown')
        timestamp += 1
        seen[middle] = ['box']
        observe()
        with self.assertRaises(RuntimeError):
            slots.confirm_placement('right', middle, 'box')
        slots.confirm_placement('left', middle, 'box')
        self.assertEqual(slots.states[middle], 'occupied')
        seen[middle] = []
        observe()
        self.assertEqual(slots.states[middle], 'occupied')
        timestamp += 1
        observe()
        self.assertEqual(slots.states[middle], 'free')
        self.assertEqual(slots.owners[right], 'right')

    def test_pair_dwell_can_wait_out_a_temporary_duplicate_in_one_slot(self):
        left,right=(-.2,-.4),(.2,-.4)
        slots=ConveyorSlots({'conveyor':{'slots_xy':[left,right]}})
        def observe(time, tracks):
            slots.observe(SimpleNamespace(slot=lambda xy:dict(free=not tracks[xy],
                observed_objects=tracks[xy],observation_time_s=time)))
        observe(1.,{left:[],right:[]})
        slots.reserve_pair({'left':left,'right':right})
        def confirmed():
            return all(slots.placement_observed(arm,xy,arm+'-box')
                       for arm,xy in [('left',left),('right',right)])
        observe(2.,{left:['left-box'],right:['right-box','fragment']})
        self.assertFalse(confirmed())
        self.assertEqual(slots.owners,{left:'left',right:'right'})
        observe(2.1,{left:['left-box'],right:['right-box']})
        self.assertTrue(confirmed())
