import unittest
from types import SimpleNamespace

from workstation.planning.task_targets import ConveyorSlots


class ConveyorSlotTests(unittest.TestCase):
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
