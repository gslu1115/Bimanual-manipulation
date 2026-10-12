import unittest
from itertools import product
import numpy as np
from tests import test_observed_workspace
from workstation.perception.rgbd_cartons import deproject


def exhaustive_reference(ws):
    """Original all-corners/all-sources predicate, without short circuiting."""
    frames=list(ws._frames())
    indices=np.indices(ws.shape).reshape(3,-1).T
    centres=ws.lower+(indices+.5)*ws.resolution
    known=np.ones(len(centres),bool)
    for corner in product((-.5,.5),repeat=3):
        points=centres+np.asarray(corner)*ws.resolution
        clear=ws._modelled(points)
        for frame in frames:clear |= ws._ray_free(points,frame)
        known &= clear
    occupied=np.zeros(ws.shape,bool)
    for frame in frames:
        points=deproject(frame)[frame.valid_depth]
        points=points[np.all((points>=ws.lower)&(points<ws.upper),axis=1)]
        if not len(points):continue
        points=points[~ws._modelled(points)]
        if not len(points):continue
        grid=np.floor((points-ws.lower)/ws.resolution).astype(int)
        occupied[tuple(grid.T)]=True
    return (~known).reshape(ws.shape),occupied


class WorkspaceShortCircuitTests(unittest.TestCase):
    def test_exact_unknown_and_occupied_maps_for_occlusion_missing_depth_and_robot(self):
        fixture=test_observed_workspace.WorkspaceTests()
        for index in range(8):
            with self.subTest(case=index):
                ws=fixture.workspace(depth=1. if index%2 else .8,valid=index!=3,
                                     status='OK' if index!=4 else 'STALE')
                frame=next(iter(ws.packet.cameras.values()))
                frame.valid_depth[10:20,15:25]=False
                frame.depth_m[25:32,18:28]=.9
                if index>=5:
                    ws.robot_links=((fixture.cube(),np.array([0.,0.,.95]),np.eye(3)),)
                expected_unknown,expected_occupied=exhaustive_reference(ws)
                ws.build()
                np.testing.assert_array_equal(ws.unknown,expected_unknown)
                np.testing.assert_array_equal(ws.occupied,expected_occupied)

    def test_failed_corner_cannot_be_rescued_by_later_visible_corner(self):
        ws=test_observed_workspace.WorkspaceTests().workspace(valid=False)
        ws._modelled=lambda points:np.zeros(len(points),bool)
        ws._ray_free=lambda points,frame:points[:,2]>.81
        expected,_=exhaustive_reference(ws)
        ws.build();np.testing.assert_array_equal(ws.unknown,expected)
        self.assertTrue(ws.unknown[:,:,0].all())
