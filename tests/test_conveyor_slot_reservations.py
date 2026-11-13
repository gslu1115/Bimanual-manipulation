import unittest
from types import SimpleNamespace

from workstation.planning.task_targets import ConveyorSlots


class ConveyorSlotTests(unittest.TestCase):
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
