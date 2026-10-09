"""Independent evaluator must measure correct side/inverted semantics."""
import math
import unittest
from types import SimpleNamespace
import numpy as np
from workstation.diagnostics.vision_check import evaluate_single_estimate
from workstation.observations.camera_geometry import quaternion_from_matrix
from workstation.task_logic import BoxState


class VisionEvaluationTests(unittest.TestCase):
    def sample(self,rotation=np.eye(3),posture='UPRIGHT',yaw=0.):
        obj=SimpleNamespace(position_m=(0.,-.1,.7825),axis_yaw_rad=yaw,posture=posture,failures=())
        box=BoxState('offline_reference',list(obj.position_m),list(quaternion_from_matrix(rotation)),[0.]*3,[0.]*3)
        return obj,box

    def evaluate(self,obj,box):
        return evaluate_single_estimate(obj,box,(.1,.055,.045))

    def test_upright_keeps_existing_geometry_and_semantic_criterion(self):
        obj,box=self.sample();result=self.evaluate(obj,box)
        self.assertTrue(result['passed']);self.assertEqual(result['reference_posture'],'UPRIGHT')

    def test_positive_inverted_estimate_is_accepted(self):
        obj,box=self.sample(np.diag([1.,-1.,-1.]),'INVERTED')
        self.assertTrue(self.evaluate(obj,box)['passed'])

    def test_side_estimate_is_accepted_without_unique_top_direction(self):
        r=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        obj,box=self.sample(r,'SIDE');result=self.evaluate(obj,box)
        self.assertTrue(result['passed']);self.assertEqual(result['reference_posture'],'SIDE')

    def test_tall_side_uses_horizontal_y_axis_instead_of_vertical_x(self):
        r=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        obj,box=self.sample(r,'SIDE',math.pi/2);result=self.evaluate(obj,box)
        self.assertTrue(result['passed']);self.assertEqual(result['reference_horizontal_long_axis'],1)

    def test_axis_symmetry_is_preserved(self):
        obj,box=self.sample(yaw=math.pi)
        self.assertTrue(self.evaluate(obj,box)['passed'])

    def test_unknown_is_not_accepted_despite_correct_geometry(self):
        obj,box=self.sample(posture='UNKNOWN');result=self.evaluate(obj,box)
        self.assertTrue(result['geometry_passed']);self.assertFalse(result['posture_passed'])
        self.assertFalse(result['passed'])

    def test_inverted_reported_upright_is_explicit_semantic_failure(self):
        obj,box=self.sample(np.diag([1.,-1.,-1.]),'UPRIGHT');result=self.evaluate(obj,box)
        self.assertTrue(result['geometry_passed']);self.assertFalse(result['posture_passed'])

    def test_position_and_yaw_thresholds_remain_independent_of_posture(self):
        obj,box=self.sample();obj.position_m=(.006,-.1,.7825)
        self.assertFalse(self.evaluate(obj,box)['geometry_passed'])
        obj.position_m=tuple(box.position);obj.axis_yaw_rad=math.radians(6)
        self.assertFalse(self.evaluate(obj,box)['geometry_passed'])

    def test_tilted_reference_is_unsupported_not_forced_to_upright(self):
        a=math.radians(15);c,s=math.cos(a),math.sin(a)
        obj,box=self.sample(np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]]))
        result=self.evaluate(obj,box)
        self.assertEqual(result['reference_posture'],'UNSUPPORTED_TILT');self.assertFalse(result['passed'])

    def test_failed_partial_geometry_cannot_pass_on_semantic_label(self):
        obj,box=self.sample();obj.failures=('PARTIAL_OR_MERGED_GEOMETRY',)
        result=self.evaluate(obj,box)
        self.assertTrue(result['posture_passed']);self.assertFalse(result['geometry_passed'])
        self.assertFalse(result['passed'])


if __name__=='__main__':unittest.main()
