import copy
import json
import math
from pathlib import Path
import unittest
import numpy as np
from handover_math import turn_y, multiply, inverse, slerp, angle_between
from task_logic import rotation, orientation_error, validate, slot_errors, BoxState, single_pnp_success


class HandoverGeometryTests(unittest.TestCase):
    def test_handover_keeps_top_up_after_receiver_turn(self):
        box = turn_y(-math.pi/2)
        for roll in ([1, 0, 0, 0], [0, 0, 0, 1]):
            receiver = multiply(multiply(box, [0, 0, 1, 0]), roll)
            relative = multiply(inverse(receiver), box)
            final_box = multiply(turn_y(0), relative)
            tilt, yaw = orientation_error(final_box)
            self.assertLess(tilt, 1e-5)
            self.assertLess(yaw, 1e-5)
            self.assertLess(np.array(rotation(box))[0, 2]*np.array(rotation(receiver))[0, 2], -.99)

    def test_slerp_quaternion_sign_and_half_turn(self):
        q = turn_y(0)
        for t in (0, .25, .5, 1):
            self.assertLess(angle_between(q, slerp(q, -q, t)), 1e-6)
        self.assertAlmostEqual(angle_between(q, slerp(q, turn_y(-math.pi), .5)), math.pi/2)

    def test_inverted_at_target_cannot_pass(self):
        c = json.loads(Path(__file__).with_name('config.json').read_text())
        xy = c['conveyor']['slots_xy'][2]
        box = BoxState('b', xy+[c['conveyor']['top_z']+c['boxes']['size'][2]/2], [0, 1, 0, 0], [0]*3, [0]*3)
        self.assertFalse(single_pnp_success(slot_errors(box, xy, c), c, .2, 1., [.04]*2, True))

    def test_reject_unsafe_fixture_configuration(self):
        original = json.loads(Path(__file__).with_name('config.json').read_text())
        validate(original)
        for key, value in [('receiver', 'panda_left'), ('grip_position', .04),
                           ('giver_offset_x', -.06), ('handover_center', [0, 0, .80]), ('slot_index', -1)]:
            c = copy.deepcopy(original)
            c['dual_handover'][key] = value
            with self.assertRaises(ValueError): validate(c)


if __name__ == '__main__': unittest.main()
