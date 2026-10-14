import unittest
import numpy as np
from tests import test_observed_workspace as fixtures


def reference(model,points,position,rotation,margin):
    local=(points-position)@rotation
    broad=np.all((local>=model.lower-margin-1e-8)&(local<=model.upper+margin+1e-8),axis=1)
    result=np.zeros(len(points),bool);indices=np.flatnonzero(broad)
    result[indices]=np.all(local[indices]@model.planes[:,:3].T+model.planes[:,3]<=margin+1e-8,axis=1)
    return result


class RobotDepthBroadphaseTests(unittest.TestCase):
    def test_exact_membership_for_rotated_dense_clouds_and_surface_roundoff(self):
        model=fixtures.WorkspaceTests().cube();rng=np.random.default_rng(928)
        for dtype in (np.float32,np.float64):
            for angle in (0.,.7,2.9):
                c,s=np.cos(angle),np.sin(angle)
                rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=dtype)
                position=np.array([.2,-.15,.9],dtype=dtype)
                for margin in (0.,.003,.008):
                    local=rng.uniform(-.1,.1,(800,3))
                    local[:100,0]=model.upper[0]+margin+1e-8+rng.uniform(-1e-7,1e-7,100)
                    points=(local@np.linalg.inv(rotation)+position).astype(dtype)
                    with self.subTest(dtype=dtype,angle=angle,margin=margin):
                        np.testing.assert_array_equal(model.contains(points,position,rotation,margin),reference(model,points,position,rotation,margin))

    def test_empty_far_and_singular_transform_follow_original_predicate(self):
        model=fixtures.WorkspaceTests().cube()
        for points in (np.empty((0,3)),np.full((600,3),10.)):
            for rotation in (np.eye(3),np.zeros((3,3)),np.diag([1.,2.,.5])):
                np.testing.assert_array_equal(model.contains(points,np.zeros(3),rotation),reference(model,points,np.zeros(3),rotation,.003))
