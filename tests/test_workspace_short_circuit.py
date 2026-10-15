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
    def test_corner_batches_keep_all_corners_and_first_failure_witness(self):
        ws=test_observed_workspace.WorkspaceTests().workspace()
        rng=np.random.default_rng(320)
        centres=np.c_[rng.uniform(-.04,.04,(5003,2)),np.full(5003,.95)]
        ws._clear_points=lambda points,frames:points[:,0]+points[:,1]>=0
        known,witness=ws._corner_evidence(centres,.005,[])
        expected=centres[:,0]+centres[:,1]-.005>=0
        np.testing.assert_array_equal(known,expected)
        np.testing.assert_array_equal(witness[~known],centres[~known]-.0025)

    def test_fine_evidence_keeps_union_of_rays_models_and_history(self):
        from types import SimpleNamespace
        fixture=test_observed_workspace.WorkspaceTests();ws=fixture.workspace()
        ws.robot_links=((fixture.cube(),np.array([0.,0.,.95]),np.eye(3)),)
        frame=next(ws._frames());frame.valid_depth[20:30,20:30]=False
        ws.occlusion_memory=SimpleNamespace(free=lambda points,current:
            (points[:,0]>.025)&(points[:,2]>.88))
        rng=np.random.default_rng(210)
        points=rng.uniform([-.08,-.08,.74],[.08,.08,1.12],(3000,3))
        expected=ws._modelled(points)
        for f in ws._frames():expected |= ws._ray_free(points,f)
        expected |= ws.occlusion_memory.free(points,ws)
        np.testing.assert_array_equal(ws._clear_points(points,list(ws._frames())),expected)

    def test_current_ray_evidence_avoids_unneeded_robot_and_history_work(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        ws=test_observed_workspace.WorkspaceTests().workspace()
        points=np.array([[0,0,.95],[.01,0,.96]])
        ws._modelled=Mock(side_effect=AssertionError('Unneeded geometry query'))
        ws.occlusion_memory=SimpleNamespace(free=Mock(side_effect=AssertionError('Unneeded history query')))
        self.assertTrue(ws._clear_points(points,list(ws._frames())).all())
        ws._modelled.assert_not_called();ws.occlusion_memory.free.assert_not_called()

    def test_modelled_union_short_circuit_preserves_overlapping_boxes_and_links(self):
        from workstation.planning.collision import AABB
        fixture=test_observed_workspace.WorkspaceTests();ws=fixture.workspace()
        ws.static=(*ws.static,AABB('overlap',(0,0,.82),(.04,.04,.04)),AABB('small',(.03,0,.93),(.02,.03,.03)))
        ws.robot_links=((fixture.cube(),np.array([0.,0.,.95]),np.eye(3)),)
        rng=np.random.default_rng(208)
        points=rng.uniform([-.08,-.08,.73],[.08,.08,1.05],(4000,3))
        expected=np.zeros(len(points),bool)
        for box in ws.static+ws.objects:
            expected|=np.all(np.abs(points-box.centre)<=np.asarray(box.size)/2+.002,axis=1)
        for model,p,r in ws.robot_links:expected|=model.contains(points,p,r)
        np.testing.assert_array_equal(ws._modelled(points),expected)

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
