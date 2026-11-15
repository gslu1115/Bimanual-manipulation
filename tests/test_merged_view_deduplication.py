import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors
from workstation.planning.collision import AABB


class MergedViewDeduplicationTests(unittest.TestCase):
    def fixture(self):
        priors=ScenePriors((.1,.05,.05),.8,(-.4,.4,-.4,.4))
        estimator=CartonEstimator(priors)
        points=np.zeros((20,10,3));points[:10]=[-.12,0,.825];points[10:]=[.12,0,.825]
        frame=NS(sample_time_s=1.,depth_m=np.ones((20,10)),valid_depth=np.ones((20,10),bool))
        masks={f'camera/{k}':np.ones((20,10),bool) for k in ('left','right','merged')}
        packet=NS(observation_time_s=1.,assembled_time_s=1.,cameras={'camera':frame},camera_status={'camera':'OK'})
        def box(name,x,failures=()):
            return NS(time_s=1.,mask_refs=(f'camera/{name}',),position_m=(x,0,.825),
                axis_yaw_rad=0.,size_m=(.1,.05,.05),uncertainty_m=.003,failures=failures,
                visibility='PARTIAL_OR_UNCERTAIN' if failures else 'COMPLETE_TOP')
        detections=[box('left',-.12),box('right',.12),box('merged',0.,('PARTIAL_OR_MERGED_GEOMETRY',))]
        return estimator,packet,masks,detections,points

    def test_union_removes_only_redundant_failed_fit_and_keeps_real_objects_and_masks(self):
        est,p,m,d,points=self.fixture()
        with patch('workstation.perception.rgbd_cartons.deproject',return_value=points):
            result=est._deduplicate_covered_views(d,m,p)
        self.assertEqual(result,d[:2]);self.assertEqual(len(m),3)

    def test_one_unexplained_point_or_missing_depth_retains_obstacle(self):
        for kind in ('outside','invalid','stale'):
            est,p,m,d,points=self.fixture()
            if kind=='outside':points[0,0]=[0,0,1.1]
            if kind=='invalid':p.cameras['camera'].valid_depth[0,0]=False
            if kind=='stale':p.assembled_time_s=2.
            with patch('workstation.perception.rgbd_cartons.deproject',return_value=points):
                self.assertEqual(est._deduplicate_covered_views(d,m,p),d)

    def test_stale_reference_or_single_object_cannot_explain_merged_view(self):
        for kind in ('stale_reference','single_reference'):
            est,p,m,d,points=self.fixture()
            if kind=='stale_reference':d[0].time_s=0.
            else:d.pop(0)
            with patch('workstation.perception.rgbd_cartons.deproject',return_value=points):
                self.assertEqual(est._deduplicate_covered_views(d,m,p),d)

    def test_unknown_independent_object_is_not_removed(self):
        est,p,m,d,points=self.fixture();d[-1].failures=('APPEARANCE_AMBIGUOUS',)
        with patch('workstation.perception.rgbd_cartons.deproject',return_value=points):
            self.assertEqual(est._deduplicate_covered_views(d,m,p),d)

    def test_known_fixture_duplicate_removed_but_one_outside_point_kept(self):
        for outside in (False,True):
            est,p,m,d,points=self.fixture();d=d[-1:]
            est.known_static=(AABB('fixture',(0,0,.825),(.4,.1,.05)),)
            if outside:points[0,0]=[0,0,.9]
            with patch('workstation.perception.rgbd_cartons.deproject',return_value=points):
                result=est._deduplicate_covered_views(d,m,p)
            self.assertEqual(result,d if outside else [])
            self.assertEqual(est.redundant_views['fixed_geometry'],0 if outside else 1)
            self.assertEqual(len(est.known_static),1)
