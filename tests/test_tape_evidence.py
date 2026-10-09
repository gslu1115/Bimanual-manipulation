"""Measured metric top pixels; positive tape evidence is not a probability."""
import unittest
import numpy as np
from workstation.perception.rgbd_cartons import CartonEstimator


def surface(xrange=(-.045,-.015),stripe=True,width=.012,offset=0.,spot=None):
    x,y=np.meshgrid(np.arange(*xrange,.0008),np.arange(-.027,.027,.0008),indexing='ij')
    xy=np.c_[x.ravel(),y.ravel()]
    brightness=np.full(len(xy),100.)
    if stripe:
        marked=np.abs(xy[:,1]-offset)<width/2
        if spot is not None:marked&=(xy[:,0]>=spot[0])&(xy[:,0]<spot[1])
        brightness[marked]=135.
    return np.repeat(brightness[:,None],3,axis=1),xy


class TapeEvidenceTests(unittest.TestCase):
    def evidence(self,rgb,xy):
        return CartonEstimator._tape_evidence(rgb,xy,[0,0],0.,.1,.055)

    def test_partial_stripe_with_same_position_flanks(self):
        rgb,xy=surface();e=self.evidence(rgb,xy)
        self.assertEqual(e['tape_observed'],1.);self.assertEqual(e['tape_partial_support'],1.)
        self.assertGreaterEqual(e['tape_paired_bins'],5)
        self.assertGreater(e['tape_paired_span_m'],.020)
        self.assertLess(e['tape_visible_span_m'],.060)

    def test_full_stripe_preserves_existing_cue(self):
        e=self.evidence(*surface(xrange=(-.05,.05)))
        self.assertEqual(e['tape_observed'],1.);self.assertEqual(e['tape_partial_support'],0.)

    def test_plain_surface_remains_unknown(self):
        self.assertEqual(self.evidence(*surface(stripe=False))['tape_observed'],0.)

    def test_short_spot_cannot_replace_long_tape(self):
        self.assertEqual(self.evidence(*surface(spot=(-.034,-.024)))['tape_observed'],0.)

    def test_broad_highlight_is_not_a_partial_stripe(self):
        self.assertEqual(self.evidence(*surface(width=.035))['tape_observed'],0.)

    def test_offset_stripe_is_not_the_known_central_marker(self):
        self.assertEqual(self.evidence(*surface(offset=.015))['tape_observed'],0.)

    def test_missing_one_flank_cannot_support_partial_marker(self):
        rgb,xy=surface();visible=xy[:,1]>-.012
        self.assertEqual(self.evidence(rgb[visible],xy[visible])['tape_observed'],0.)

    def test_unobserved_gap_is_not_filled(self):
        rgb,xy=surface(xrange=(-.05,.0));visible=(xy[:,0]<-.034)|(xy[:,0]>-.016)
        self.assertEqual(self.evidence(rgb[visible],xy[visible])['tape_observed'],0.)

    def test_one_bright_flank_prevents_partial_contrast(self):
        rgb,xy=surface();rgb[xy[:,1]>.016]=155.
        self.assertEqual(self.evidence(rgb,xy)['tape_observed'],0.)

    def test_invalid_observations_do_not_support_tape(self):
        rgb,xy=surface();rgb[0,0]=float('nan')
        self.assertEqual(self.evidence(rgb,xy)['tape_observed'],0.)
        self.assertEqual(self.evidence(np.ones((60,3)),np.ones((60,3)))['tape_observed'],0.)

    def test_yaw_and_translation_only_change_coordinates(self):
        rgb,xy=surface();yaw=1.78;c,s=np.cos(yaw),np.sin(yaw)
        rotation=np.array([[c,-s],[s,c]]);centre=np.array([-.18,-.07])
        e=CartonEstimator._tape_evidence(rgb,xy@rotation.T+centre,centre,yaw,.1,.055)
        self.assertEqual(e['tape_observed'],1.)


class FlatPackagingTests(unittest.TestCase):
    def view(self,fold=None,top_tape=False):
        top_rgb,xy=surface(xrange=(-.05,.05),stripe=top_tape)
        top=np.c_[xy,np.full(len(xy),.805)]
        y,z=np.meshgrid(np.linspace(-.0275,.0275,40),np.linspace(-.0225,.0225,35))
        face=np.c_[np.full(y.size,.05),y.ravel(),.7825+z.ravel()]
        rgb=np.full((len(face),3),100.)
        if fold:
            band=np.abs(z.ravel())>.0045 if fold=='BOTH' else z.ravel()*(1 if fold=='UPRIGHT' else -1)>.0045
            rgb[(np.abs(y.ravel())<=.006)&band]=135.
        return top_rgb,xy,np.concatenate([top,face]),np.concatenate([top_rgb,rgb])

    def estimate(self,views):
        return CartonEstimator._flat_packaging(views,np.zeros(2),0.,.1,.055,.045,.805)

    def test_positive_upper_fold_resolves_shadowed_top(self):
        label,q,source=self.estimate({'scene_camera':self.view('UPRIGHT')})
        self.assertEqual(label,'UPRIGHT');self.assertEqual(q['tape_observed'],0.)
        self.assertEqual(q['tape_fold_upright_views'],1.);self.assertIn('scene_camera',source)

    def test_positive_lower_fold_is_inverted_evidence(self):
        label,q,_=self.estimate({'scene_camera':self.view('INVERTED')})
        self.assertEqual(label,'INVERTED');self.assertEqual(q['tape_fold_inverted_views'],1.)
        self.assertEqual({h.label for h in CartonEstimator._flat_hypotheses(0,label,'measured fold')},{'INVERTED'})

    def test_missing_marker_retains_both_hypotheses(self):
        label,_,_=self.estimate({'scene_camera':self.view()})
        self.assertEqual(label,'UNKNOWN')
        self.assertEqual({h.label for h in CartonEstimator._flat_hypotheses(0,label,'no marker')},{'UPRIGHT','INVERTED'})

    def test_conflicting_top_and_lower_fold_remain_unknown(self):
        label,q,_=self.estimate({'scene_camera':self.view('INVERTED',True)})
        self.assertEqual(label,'UNKNOWN');self.assertEqual(q['packaging_conflict'],1.)
        label,q,_=self.estimate({'scene_camera':self.view('UPRIGHT'),'left_wrist_camera':self.view('INVERTED')})
        self.assertEqual(label,'UNKNOWN');self.assertEqual(q['packaging_conflict'],1.)

    def test_merged_body_outliers_cannot_supply_fold_direction(self):
        top_rgb,xy,body,rgb=self.view('UPRIGHT')
        body=np.concatenate([body,np.full((len(body),3),.2)])
        rgb=np.concatenate([rgb,np.full_like(rgb,170.)])
        label,q,_=self.estimate({'scene_camera':(top_rgb,xy,body,rgb)})
        self.assertEqual(label,'UNKNOWN');self.assertEqual(q['tape_fold_upright_views'],0.)


if __name__=='__main__':unittest.main()
