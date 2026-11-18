import unittest
import numpy as np

from workstation.planning.pair_clock import model_clock


class PairClockTests(unittest.TestCase):
    def test_clock_has_one_final_rounding_and_preserves_delay(self):
        times=np.linspace(0,.83,84)
        points=np.zeros((len(times),2,7))
        points[:,0,0]=times*.2
        points[:,1,0]=np.maximum(times-.4,0)*.2
        dt=1/120
        output=model_clock(points,dt,2,100,100000,times)
        self.assertEqual(len(output)-1,int(np.ceil(times[-1]/dt)))
        np.testing.assert_allclose(output[-1],points[-1])
        self.assertTrue(np.all(output[:48,1]==0))

    def test_model_and_drive_bounds_apply_to_each_arm_at_corners(self):
        points=np.zeros((4,2,7))
        points[1:,0,0]=.4
        points[2:,1,1]=-.3
        points[3,0,0]=0
        velocity=np.full((2,7),2.)
        velocity[1,1]=.45
        dt=.01
        samples=model_clock(points,dt,velocity,4,1000)
        np.testing.assert_allclose(samples[[0,-1]],points[[0,-1]])
        v=np.diff(samples,axis=0)/dt
        a=np.diff(np.pad(v,((1,1),(0,0),(0,0))),axis=0)/dt
        j=np.diff(np.pad(a,((1,1),(0,0),(0,0))),axis=0)/dt
        self.assertTrue((abs(v)<=velocity+1e-9).all())
        self.assertLessEqual(abs(a).max(),4+1e-9)
        self.assertLessEqual(abs(j).max(),1000+1e-9)

    def test_hold_and_invalid_limits(self):
        points=np.zeros((2,2,7))
        np.testing.assert_equal(model_clock(points,.01,1,2,3),points[:1])
        with self.assertRaises(ValueError):model_clock(points,.01,0,2,3)
