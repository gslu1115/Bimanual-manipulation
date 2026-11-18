import unittest
from types import SimpleNamespace
from unittest.mock import patch

from workstation.planning import common_lift


class CommonLiftTests(unittest.TestCase):
    def setUp(self):
        self.specs=[SimpleNamespace(candidate=SimpleNamespace(arm=a,candidate_id=a),slot=(i,0))
                    for i,a in enumerate(('left','right'))]
        self.paths=['left path','right path']

    def test_boundary_keeps_original_paths_without_an_ik_search(self):
        with patch.object(common_lift,'mutual_gap',return_value=.008), patch.object(common_lift,'checked_variants') as ik:
            specs,paths,_=next(common_lift.common_lift_options(None,None,self.specs,self.paths,{},{}))
            self.assertEqual((specs,paths),(self.specs,self.paths))
            ik.assert_not_called()

    def test_close_lift_preserves_candidate_and_uses_checked_alternate_path(self):
        alternate=SimpleNamespace(candidate=self.specs[0].candidate,slot=(0,0))
        with patch.object(common_lift,'mutual_gap',side_effect=[.00435,.018]), patch.object(
                common_lift,'checked_variants',return_value=[(alternate,'new left path',.02)]):
            specs,paths,detail=next(common_lift.common_lift_options(None,None,self.specs,self.paths,{},{}))
        self.assertEqual(specs,(alternate,self.specs[1]))
        self.assertEqual(paths,('new left path','right path'))
        self.assertEqual(detail['outward_arms'],['left'])

    def test_exhausted_postures_return_to_existing_candidate_loop(self):
        with patch.object(common_lift,'mutual_gap',return_value=.00799), patch.object(
                common_lift,'checked_variants',return_value=[]):
            self.assertEqual(list(common_lift.common_lift_options(None,None,self.specs,self.paths,{},{})),[])
