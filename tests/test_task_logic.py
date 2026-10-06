import copy
import json
import math
import unittest
from pathlib import Path

from workstation.task_logic import BoxState, IndexingConveyor, TaskLoop, orientation_error, release_states, single_pnp_success, slot_errors, slot_valid, validate


class TaskLogicTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((Path(__file__).resolve().parents[1]/'config'/'scene.json').read_text())

    def batch(self):
        return [BoxState(f'box_{i:02d}', xy+[self.c['conveyor']['top_z']+self.c['boxes']['size'][2]/2],
                         [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
                for i, xy in enumerate(self.c['conveyor']['slots_xy'])]

    def test_yaw_symmetry_and_inverted_rejection(self):
        self.assertEqual(orientation_error([0, 0, 0, 1]), (0, 0))
        self.assertAlmostEqual(orientation_error([math.sqrt(.5), 0, 0, math.sqrt(.5)])[1], 90)
        b = self.batch()[0]
        b.orientation_wxyz = [0, 1, 0, 0]
        self.assertFalse(slot_valid(slot_errors(b, self.c['conveyor']['slots_xy'][0], self.c), self.c))

    def test_dwell_does_not_count_transient_or_identity_switch(self):
        conv = IndexingConveyor(self.c)
        boxes = self.batch()
        for _ in range(40): conv.update(boxes, .01)
        self.assertEqual(conv.state, 'WAITING')
        boxes[0].name = 'replacement'
        for _ in range(20): conv.update(boxes, .01)
        self.assertEqual(conv.state, 'WAITING')
        for _ in range(40): conv.update(boxes, .01)
        self.assertEqual(conv.state, 'INDEXING')

    def test_conveyor_needs_three_and_clearance(self):
        conv = IndexingConveyor(self.c)
        boxes = self.batch()
        for _ in range(100): conv.update(boxes[:2], .01)
        self.assertEqual(conv.state, 'WAITING')
        for _ in range(100): conv.update(boxes, .01)
        self.assertEqual(conv.state, 'INDEXING')
        for _ in range(500): conv.update(boxes, .01)
        self.assertEqual(conv.completed_batches, 0)  # clock alone cannot create success
        for b in boxes: b.position[0] += .7
        conv.update(boxes, .01)
        self.assertEqual(conv.completed_batches, 1)

    def test_blocker_and_safety_interlock(self):
        boxes = self.batch()
        conv = IndexingConveyor(self.c)
        for _ in range(100): conv.update(boxes, .01, safe_to_index=False)
        self.assertEqual(conv.state, 'WAITING')
        extra = copy.deepcopy(boxes[0])
        extra.name = 'stacked'
        extra.position[2] += self.c['boxes']['size'][2]
        for _ in range(100): conv.update(boxes+[extra], .01)
        self.assertEqual(conv.state, 'WAITING')
        conv.update(boxes, .01)
        self.assertEqual(conv.state, 'INDEXING')
        self.assertEqual(conv.update(boxes, .01, False), 0.)
        self.assertEqual(conv.state, 'FAULT')

    def test_direct_priority_and_failed_action_reobserve(self):
        loop = TaskLoop()
        self.assertEqual(loop.decide(['a'], ['a']), 'DIRECT_PNP')
        loop.action_finished()
        loop.verified(False)
        self.assertEqual(loop.decide(['a'], ['a']), 'DIRECT_PNP')
        loop.action_finished()
        loop.verified(True)
        self.assertEqual(loop.decide(['a'], []), 'UNLOCK')
        loop.action_finished()
        loop.verified(True)
        self.assertEqual(loop.decide([], []), 'DONE')

    def test_release_reproducibility_and_nonoverlap(self):
        validate(self.c)
        self.assertEqual(release_states(self.c, 6), release_states(self.c, 6))
        self.assertNotEqual(release_states(self.c, 6), release_states(self.c, 7))
        boxes = release_states(self.c, 6)
        radius = math.sqrt(sum(v*v for v in self.c['boxes']['size']))/2
        for i, a in enumerate(boxes):
            for b in boxes[i+1:]:
                self.assertGreater(math.dist(a.position, b.position), 2*radius)

    def test_single_box_config_accepts_legacy_and_current_scenes(self):
        validate(self.c)
        legacy = copy.deepcopy(self.c)
        legacy.pop('single_pick_place', None)
        validate(legacy)

    def test_single_box_config_rejects_unusable_robot_and_gripper_parameters(self):
        changes = [('arm', 'missing_arm'), ('slot_index', True), ('slot_index', 3),
                   ('box_xy', [0.0]), ('box_yaw_deg', float('nan')),
                   ('lift_height', 0), ('move_speed', -1), ('joint_speed', 'fast'),
                   ('grip_position', .04), ('grip_position', .03),
                   ('grip_position', -.001), ('grip_max_force', 0),
                   ('verify_timeout_s', self.c['verification']['hold_s']/2)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                c = copy.deepcopy(self.c)
                c['single_pick_place'][key] = value
                with self.assertRaises(ValueError): validate(c)

    def test_single_box_config_checks_rotated_footprint_not_only_center(self):
        c = copy.deepcopy(self.c)
        s, u = c['single_pick_place'], c['upstream']
        # At 90 degrees this narrow side fits; rotating to 0 degrees intersects
        # the rim even though the center remains inside the nominal tray.
        s['box_xy'] = [u['center_xy'][0]+u['tray_size_xy'][0]/2-.04, u['center_xy'][1]]
        s['box_yaw_deg'] = 90
        validate(c)
        s['box_yaw_deg'] = 0
        with self.assertRaises(ValueError): validate(c)

    def test_single_box_success_requires_physical_lift_release_dwell_and_retreat(self):
        b = self.batch()[0]
        errors = slot_errors(b, self.c['conveyor']['slots_xy'][0], self.c)
        evidence = dict(lift_m=.10, stable_s=.6, finger_positions=[.04, .04], arm_home=True)
        self.assertTrue(single_pnp_success(errors, self.c, **evidence))
        changes = [('lift_m', 0), ('lift_m', .079), ('lift_m', float('inf')),
                   ('stable_s', .49), ('stable_s', float('nan')),
                   ('finger_positions', [.04, .025]), ('finger_positions', [.037, .04]),
                   ('finger_positions', [.04]), ('arm_home', False)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                bad = {**evidence, key: value}
                self.assertFalse(single_pnp_success(errors, self.c, **bad))
        b.position[0] += self.c['verification']['xy_tolerance']+.001
        self.assertFalse(single_pnp_success(slot_errors(b, self.c['conveyor']['slots_xy'][0], self.c), self.c, **evidence))


if __name__ == '__main__': unittest.main()
