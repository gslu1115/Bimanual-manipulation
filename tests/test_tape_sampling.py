import unittest
import numpy as np
from workstation.perception.packaging_evidence import tape_axis_evidence


class TapeSamplingTests(unittest.TestCase):
    def view(self,sign):
        parts=[]
        for x in np.arange(-.03,.034,.004):
            y=np.linspace(-.027,.027,145)
            # Unequal image sampling across a physically straight stripe.
            dense=np.repeat(y[(abs(y)<.0058)&(abs(y-sign*.09*x)<.001)],10)
            y=np.r_[y,dense]
            parts.append(np.column_stack((np.full(len(y),x),y,np.full(len(y),.0225))))
        points=np.concatenate(parts);rgb=np.tile([140,95,55],(len(points),1))
        rgb[abs(points[:,1])<.006]=[215,205,170]
        return points,rgb

    def measure(self,views):
        return tape_axis_evidence(views,np.zeros(3),np.eye(3),np.array([0.,0.,1.]),np.array([.1,.055,.045]))

    def test_opposite_sampling_biases_preserve_one_measured_direction(self):
        result=self.measure([self.view(-1),self.view(1)])
        self.assertIsNotNone(result)
        self.assertEqual(result['camera_count'],2)
        self.assertLess(abs(result['axis_workcell'][1]),np.sin(np.deg2rad(1.5)))

    def test_real_cross_view_direction_conflict_is_still_rejected(self):
        p,rgb=self.view(1);a=np.deg2rad(12);c,s=np.cos(a),np.sin(a)
        turn=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        self.assertIsNone(self.measure([self.view(-1),(p@turn.T,rgb)]))

    def test_dense_plain_cardboard_does_not_create_a_tape_direction(self):
        p,rgb=self.view(1);rgb[:]=[140,95,55]
        self.assertIsNone(self.measure([(p,rgb)]))


if __name__=='__main__':unittest.main()
