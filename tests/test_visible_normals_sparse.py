import unittest
import numpy as np
from workstation.observations.camera_geometry import visible_surface_normals


def dense_reference(xyz,visible):
    du=xyz[1:-1,2:]-xyz[1:-1,:-2];dv=xyz[2:,1:-1]-xyz[:-2,1:-1]
    cross=np.cross(du,dv);length=np.linalg.norm(cross,axis=-1)
    interior=(visible[1:-1,1:-1]&visible[1:-1,2:]&visible[1:-1,:-2]&
        visible[2:,1:-1]&visible[:-2,1:-1]&(length>1e-12)&
        (np.linalg.norm(du,axis=-1)<.008)&(np.linalg.norm(dv,axis=-1)<.008))
    rows,cols=np.nonzero(interior)
    return rows+1,cols+1,xyz[1:-1,1:-1][interior],cross[interior]/length[interior,None]


class VisibleNormalsSparseTests(unittest.TestCase):
    def assert_same(self,xyz,visible):
        for actual,expected in zip(visible_surface_normals(xyz,visible),dense_reference(xyz,visible)):
            np.testing.assert_array_equal(actual,expected)

    def test_exact_normals_for_sparse_occluded_depth_and_jumps(self):
        rng=np.random.default_rng(19)
        v,u=np.indices((61,83))
        xyz=np.stack((u*.001,v*.001,.8+.0001*u+.0003*v),axis=-1)
        xyz+=rng.normal(0,.0001,xyz.shape)
        xyz[20:30,35:45,2]+=.04
        visible=np.zeros((61,83),bool);visible[3:52,7:71]=True
        visible[rng.random(visible.shape)<.04]=False
        self.assert_same(xyz,visible)

    def test_empty_degenerate_and_image_boundaries(self):
        for shape in ((2,2),(5,7),(20,25)):
            xyz=np.zeros((*shape,3))
            for visible in (np.zeros(shape,bool),np.ones(shape,bool)):
                self.assert_same(xyz,visible)

    def test_strict_depth_jump_threshold_and_no_wrapped_neighbors(self):
        v,u=np.indices((9,11))
        for spacing in (.004,np.nextafter(.004,0.),np.nextafter(.004,1.)):
            xyz=np.stack((u*spacing,v*.001,np.ones(u.shape)),axis=-1)
            self.assert_same(xyz,np.ones(u.shape,bool))
