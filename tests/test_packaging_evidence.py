import unittest
import numpy as np
from workstation.perception.packaging_evidence import tape_fold_evidence


class PackagingEvidenceTests(unittest.TestCase):
    def face(self,orientation=None):
        y,z=np.meshgrid(np.linspace(-.0275,.0275,40),np.linspace(-.0225,.0225,35))
        points=np.column_stack((np.full(y.size,.05),y.ravel(),.7825+z.ravel()))
        rgb=np.tile([140,95,55],(y.size,1)).astype(np.uint8)
        if orientation:
            direction=1 if orientation=='UPRIGHT' else -1
            tape=(np.abs(y.ravel())<=.006)&(z.ravel()*direction>.0045)
            rgb[tape]=[170,125,85]
        return points,rgb

    def evaluate(self,points,rgb):
        return tape_fold_evidence(points,rgb,(0,0),0,.1,.055,.045,.805)

    def test_positive_upper_fold(self):
        self.assertEqual(self.evaluate(*self.face('UPRIGHT'))['label'],'UPRIGHT')

    def test_positive_lower_fold(self):
        self.assertEqual(self.evaluate(*self.face('INVERTED'))['label'],'INVERTED')

    def test_plain_face_does_not_prove_inversion(self):
        self.assertEqual(self.evaluate(*self.face())['label'],'UNKNOWN')

    def test_partial_tiny_patch_does_not_prove_direction(self):
        p,c=self.face('INVERTED');self.assertEqual(self.evaluate(p[:10],c[:10])['label'],'UNKNOWN')

    def test_visible_top_and_bottom_folds_remain_ambiguous(self):
        p,a=self.face('UPRIGHT');_,b=self.face('INVERTED')
        self.assertTrue(self.evaluate(p,np.maximum(a,b))['conflicting'])
