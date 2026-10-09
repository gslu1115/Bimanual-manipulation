"""Holding-surface identity evidence cannot fabricate geometry or hide peers."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.perception.scene_estimate import ScenePriors


class SupportedTargetTests(unittest.TestCase):
    def setUp(self):
        self.est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)))
        self.scene=self.est.estimate(packet(yaw=0.));obj=self.scene.objects[0]
        self.target=obj.track_id
        self.alias=replace(obj,track_id='visual_alias',position_m=(0.,-.1,.93),
            posture='UNKNOWN',failures=('VERTICAL_EXTENT_UNRESOLVED',),
            visibility='PARTIAL_OR_UNCERTAIN',velocity_m_s=(0.,0.,1.))
        self.scene=replace(self.scene,objects=(self.alias,))
        self.proof=dict(time_s=1.,mask_refs=list(self.alias.mask_refs),surface_inlier_ratio=.97,
            top_coverage=.5,side_points=80,plane_residual_m=.001,surface_tilt_deg=1.)

    def test_supported_alias_reuses_id_without_completing_its_pose(self):
        self.est._tracks={self.alias.track_id:self.alias};self.est._anchors[self.alias.track_id]=self.alias
        result=self.est.associate_supported_target(self.scene,self.target,self.proof)
        obj=result.objects[0]
        self.assertEqual(obj.track_id,self.target)
        self.assertEqual(obj.position_m,self.alias.position_m)
        self.assertEqual(obj.posture,'UNKNOWN');self.assertEqual(obj.failures,self.alias.failures)
        self.assertEqual(obj.visibility,'PARTIAL_OR_UNCERTAIN');self.assertIsNone(obj.velocity_m_s)
        self.assertIs(result.masks,self.scene.masks)
        self.assertNotIn(self.alias.track_id,self.est._tracks)
        self.assertNotIn(self.alias.track_id,self.est._anchors)
        self.assertIs(self.est._tracks[self.target],obj)
        self.assertNotEqual(self.est._anchors[self.target].position_m,obj.position_m)

    def test_peer_with_unproved_mask_stays_an_independent_obstacle(self):
        ref='scene_camera/100/peer'
        peer=replace(self.alias,track_id='visual_peer',mask_refs=(ref,),position_m=(.2,-.1,.7825))
        masks=dict(self.scene.masks);masks[ref]=next(iter(masks.values()))
        scene=replace(self.scene,objects=(self.alias,peer),masks=masks)
        result=self.est.associate_supported_target(scene,self.target,self.proof)
        self.assertEqual([obj.track_id for obj in result.objects],[self.target,'visual_peer'])
        self.assertIs(result.objects[1],peer)

    def test_old_or_incomplete_surface_measurement_does_not_relabel(self):
        changes=[{'time_s':.9},{'time_s':float('nan')},{'surface_inlier_ratio':.94},
            {'top_coverage':.29},{'side_points':19},{'plane_residual_m':.003},
            {'surface_tilt_deg':9},{'mask_refs':[]},{'mask_refs':['scene_camera/missing']}]
        for change in changes:
            proof=dict(self.proof,**change)
            self.assertIs(self.est.associate_supported_target(self.scene,self.target,proof),self.scene)

    def test_partially_supported_multi_view_instance_is_not_absorbed(self):
        alias=replace(self.alias,mask_refs=self.alias.mask_refs+('left_wrist_camera/100/unproved',))
        scene=replace(self.scene,objects=(alias,))
        self.assertIs(self.est.associate_supported_target(scene,self.target,self.proof),scene)

    def test_original_target_separately_visible_is_a_conflict(self):
        original=replace(self.alias,track_id=self.target,mask_refs=('scene_camera/100/original',))
        scene=replace(self.scene,objects=(self.alias,original))
        with self.assertRaisesRegex(RuntimeError,'HELD_TARGET_ASSOCIATION_CONFLICT'):
            self.est.associate_supported_target(scene,self.target,self.proof)

    def test_already_matching_identity_is_not_rewritten(self):
        original=replace(self.alias,track_id=self.target)
        scene=replace(self.scene,objects=(original,))
        self.assertIs(self.est.associate_supported_target(scene,self.target,self.proof),scene)

    def test_reconciled_partial_track_stays_matchable_on_next_observation(self):
        result=self.est.associate_supported_target(self.scene,self.target,self.proof)
        obj=result.objects[0]
        self.est._detect=lambda frame,segmentation=None:([replace(obj,track_id='',time_s=1.2)],self.scene.masks)
        current=self.est.estimate(packet(time=1.2,yaw=0.))
        self.assertEqual(current.objects[0].track_id,self.target)
        self.assertIsNone(current.objects[0].velocity_m_s)


if __name__=='__main__':unittest.main()
