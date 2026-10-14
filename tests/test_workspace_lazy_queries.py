import unittest
import numpy as np
from tests import test_observed_workspace as fixtures
from tests.test_workspace_short_circuit import exhaustive_reference


class LazyWorkspaceTests(unittest.TestCase):
    def test_rotated_payload_queries_match_full_grid_with_robot_and_missing_depth(self):
        fixture=fixtures.WorkspaceTests()
        lazy=fixture.workspace();full=fixture.workspace()
        for ws in (lazy,full):
            frame=next(ws._frames())
            frame.depth_m[12:20,12:20]=.82
            frame.valid_depth[30:35,20:28]=False
            ws.robot_links=((fixture.cube(),np.array([0.,0.,.95]),np.eye(3)),)
        full.build();rng=np.random.default_rng(419)
        for i in range(30):
            centre=np.r_[rng.uniform(-.055,.055,2),rng.uniform(.82,1.15)]
            size=rng.uniform(.015,.04,3)
            angle=rng.uniform(-np.pi,np.pi);c,s=np.cos(angle),np.sin(angle)
            rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            with self.subTest(query=i):
                self.assertEqual(lazy.check_oriented_box(centre,size,rotation),full.check_oriented_box(centre,size,rotation))
        lazy.build()
        np.testing.assert_array_equal(lazy.unknown,full.unknown)
        np.testing.assert_array_equal(lazy.occupied,full.occupied)

    def test_local_queries_match_full_predicate_then_full_grid_matches(self):
        fixture=fixtures.WorkspaceTests()
        for depth,valid in ((1.,True),(.8,True),(.94,True),(1.,False)):
            with self.subTest(depth=depth,valid=valid):
                lazy=fixture.workspace(depth=depth,valid=valid)
                full=fixture.workspace(depth=depth,valid=valid)
                expected_unknown,expected_occupied=exhaustive_reference(full)
                full.build()
                queries=(('check_box',((0,0,.95),(.02,.02,.02))),
                         ('check_sphere',((.02,.02,.87),.006)),
                         ('check_oriented_box',((0,0,1.08),(.03,.02,.025),np.eye(3))))
                for name,args in queries:
                    self.assertEqual(getattr(lazy,name)(*args),getattr(full,name)(*args))
                self.assertFalse(lazy._unknown_evaluated.all())
                self.assertTrue(lazy.unknown[~lazy._unknown_evaluated].all())
                lazy.build()
                self.assertTrue(lazy._unknown_evaluated.all())
                np.testing.assert_array_equal(lazy.unknown,expected_unknown)
                np.testing.assert_array_equal(lazy.occupied,expected_occupied)

    def test_obstacle_only_query_keeps_global_obstacles_without_free_grid(self):
        ws=fixtures.WorkspaceTests().workspace(depth=.94)
        self.assertEqual(ws.check_observed_sphere_obstacles((0,0,.86),.02),'OBSERVED_PATH_OBSTACLE')
        self.assertFalse(ws._unknown_evaluated.any())
        self.assertTrue(ws.unknown.all())
        self.assertTrue(ws.occupied.any())

    def test_repeated_local_query_reuses_only_same_frame_coarse_evidence(self):
        ws=fixtures.WorkspaceTests().workspace();counts=[]
        original=ws._ray_free
        def measured(points,frame):
            counts.append(len(points));return original(points,frame)
        ws._ray_free=measured
        self.assertIsNone(ws.check_box((0,0,.95),(.02,.02,.02)))
        first=sum(counts)
        self.assertGreater(first,0)
        self.assertIsNone(ws.check_box((0,0,.95),(.02,.02,.02)))
        self.assertEqual(sum(counts),first)
        ws.check_box((0,0,1.10),(.02,.02,.02))
        self.assertGreater(sum(counts),first)

    def test_invalid_new_payload_support_invalidates_all_cached_grid_evidence(self):
        ws=fixtures.WorkspaceTests().workspace();ws.build()
        self.assertFalse(ws.confirm_payload('target',[0,0,.95],np.eye(3),[.04,.02,.02],{},uncertainty_m=0.))
        self.assertIsNone(ws.unknown);self.assertIsNone(ws.occupied)
        self.assertIsNone(ws._unknown_evaluated)
