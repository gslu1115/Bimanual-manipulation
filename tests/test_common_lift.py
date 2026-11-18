import unittest
from types import SimpleNamespace
from unittest.mock import patch

from workstation.planning import common_lift


class CommonLiftTests(unittest.TestCase):
    def setUp(self):
        self.specs=[SimpleNamespace(candidate=SimpleNamespace(arm=a,candidate_id=a),slot=(i,0),lift_q=-.02)
                    for i,a in enumerate(('left','right'))]
        offsets=patch.object(common_lift,'elbow_offset',side_effect=lambda robot,arm,q,partner:q)
        offsets.start()
        self.addCleanup(offsets.stop)
        self.paths=['left path','right path']

    def test_boundary_keeps_original_paths_without_an_ik_search(self):
        with patch.object(common_lift,'mutual_gap',return_value=.008), patch.object(common_lift,'checked_variants') as ik:
            specs,paths,_=next(common_lift.common_lift_options(None,None,self.specs,self.paths,{},{}))
            self.assertEqual((specs,paths),(self.specs,self.paths))
            ik.assert_not_called()

    def test_close_lift_preserves_candidate_and_uses_checked_alternate_path(self):
        alternate=SimpleNamespace(candidate=self.specs[0].candidate,slot=(0,0),lift_q=.03)
        with patch.object(common_lift,'mutual_gap',side_effect=[.00435,.018]), patch.object(
                common_lift,'checked_variants',side_effect=[[(alternate,'new left path',.02)],[]]):
            specs,paths,detail=next(common_lift.common_lift_options(None,None,self.specs,self.paths,{},{}))
        self.assertEqual(specs,(alternate,self.specs[1]))
        self.assertEqual(paths,('new left path','right path'))
        self.assertEqual(detail['outward_arms'],['left'])

    def test_both_elbows_outside_rank_first_even_if_unilateral_gap_is_larger(self):
        choices=[[(SimpleNamespace(candidate=s.candidate,slot=s.slot,lift_q=.025),'alternate',.045)]
                 for s in self.specs]
        with patch.object(common_lift,'checked_variants',side_effect=choices),patch.object(
                common_lift,'mutual_gap',side_effect=[.00435,.04,.05,.019]):
            _,_,detail=next(common_lift.common_lift_options(None,None,self.specs,self.paths,{},{}))
        self.assertEqual(detail['selected_elbow_offsets_m'],[.025,.025])
        self.assertEqual(detail['selected_gap_m'],.019)

    def test_exhausted_postures_return_to_existing_candidate_loop(self):
        with patch.object(common_lift,'mutual_gap',return_value=.00799), patch.object(
                common_lift,'checked_variants',return_value=[]):
            self.assertEqual(list(common_lift.common_lift_options(None,None,self.specs,self.paths,{},{})),[])
