"""Current RGB-D proof may reconcile identity but cannot fill hidden geometry."""
import unittest
from dataclasses import replace
import numpy as np
from tests.test_cuboid_pose import synthetic_observation, CENTRE, PRIORS
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator, deproject
from workstation.perception.scene_estimate import PoseHypothesis


class MeasuredFragmentIdentityTests(unittest.TestCase):
    def setUp(self):
        self.est=CartonEstimator(PRIORS)
        base=self.est.estimate(packet(yaw=0.)).objects[0]
        self.anchor=replace(base,time_s=.925,position_m=tuple(CENTRE),axis_yaw_rad=0.,
            hypotheses=(PoseHypothesis('UPRIGHT',(1.,0.,0.,0.),'previous measured upright'),))
        self.packet,self.scene=synthetic_observation(np.eye(3))
        objects=[]
        for i,ref in enumerate(self.scene.masks):
            objects.append(replace(base,track_id=base.track_id if i==0 else 'alias_'+str(i),
                position_m=tuple(CENTRE+[.025*i,0.,-.025]),mask_refs=(ref,),
                cameras=(ref.split('/')[0],),posture='UNKNOWN',velocity_m_s=None,
                failures=('PARTIAL_OR_MERGED_GEOMETRY',),visibility='PARTIAL_OR_UNCERTAIN'))
        self.scene=replace(self.scene,objects=tuple(objects))
        self.est._anchors={self.anchor.track_id:self.anchor}

    def associate(self,packet=None,scene=None,anchors=None):
        return self.est._associate_measured_fragments(packet or self.packet,scene or self.scene,
            [self.anchor] if anchors is None else anchors)

    def test_current_planes_join_ids_without_new_pose_or_semantic_claim(self):
        result=self.associate()
        self.assertEqual(len(result.objects),1)
        obj=result.objects[0]
        self.assertEqual(obj.track_id,self.anchor.track_id)
        self.assertEqual(obj.position_m,self.scene.objects[0].position_m)
        self.assertEqual(obj.posture,'UNKNOWN');self.assertTrue(obj.failures)
        self.assertEqual(obj.visibility,'PARTIAL_OR_UNCERTAIN');self.assertIsNone(obj.velocity_m_s)
        self.assertIs(result.masks,self.scene.masks)
        self.assertEqual(set(obj.mask_refs),set(self.scene.masks))
        self.assertIs(self.est._anchors[self.anchor.track_id],self.anchor)
        self.assertEqual(self.anchor.time_s,.925)

    def test_unobserved_axis_does_not_inherit_previous_box_extent(self):
        # Hide BOTH signed end faces as well as the full long-axis span.
        # One visible signed end plane would legitimately constrain this axis.
        masks={ref:mask&(np.abs(deproject(self.packet.cameras[ref.split('/')[0]])[...,0]-CENTRE[0])<.040)
               for ref,mask in self.scene.masks.items()}
        scene=replace(self.scene,masks=masks)
        self.assertIs(self.associate(scene=scene),scene)

    def test_expired_hint_or_stale_packet_cannot_authorize_relabelling(self):
        self.assertIs(self.associate(anchors=[replace(self.anchor,time_s=-2.01)]),self.scene)
        self.assertIs(self.associate(packet=replace(self.packet,assembled_time_s=1.2)),self.scene)

    def test_distinct_nearby_previous_carton_keeps_identity_ambiguous(self):
        peer=replace(self.anchor,track_id='previous_peer',position_m=tuple(CENTRE+[.012,0,0]))
        self.assertIs(self.associate(anchors=[self.anchor,peer]),self.scene)

    def test_partially_supported_original_instance_cannot_absorb_aliases(self):
        original=self.scene.objects[0]
        original=replace(original,mask_refs=original.mask_refs+('scene_camera/100/unproved',))
        scene=replace(self.scene,objects=(original,)+self.scene.objects[1:])
        self.assertIs(self.associate(scene=scene),scene)

    def test_whole_unsupported_mask_stays_an_obstacle(self):
        ref=list(self.scene.masks)[-1];mask=self.scene.masks[ref].copy()
        # More than 5% background depth cannot be cropped into target evidence.
        mask[:35,:]=True
        scene=replace(self.scene,masks={**self.scene.masks,ref:mask})
        result=self.associate(scene=scene)
        peer=scene.objects[-1]
        self.assertIn(peer,result.objects)
        self.assertEqual(next(o for o in result.objects if o.track_id==peer.track_id).mask_refs,(ref,))
        self.assertIs(result.masks,scene.masks)
