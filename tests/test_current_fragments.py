"""Current RGB-D surface association, independent of clipped fitted centres."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator,deproject
from workstation.perception.scene_estimate import ScenePriors


class CurrentFragmentTests(unittest.TestCase):
    def setUp(self):
        self.est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)))
        self.packet=packet(yaw=0.)
        frame=self.packet.cameras['scene_camera']
        self.anchor=self.est.estimate(self.packet).objects[0]
        self.full_mask=frame.depth_m<.72
        fragment=self.full_mask.copy();fragment[:,155:]=False
        ref='left_wrist_camera/100/partial'
        cameras=dict(self.packet.cameras);cameras['left_wrist_camera']=replace(frame,name='left_wrist_camera')
        status=dict(self.packet.camera_status);status['left_wrist_camera']='OK'
        self.packet=replace(self.packet,cameras=cameras,camera_status=status)
        self.fragment=replace(self.anchor,track_id='',position_m=(-.03,-.1,.755),
            mask_refs=(ref,),cameras=('left_wrist_camera',),posture='UNKNOWN',
            failures=('PARTIAL_OR_MERGED_GEOMETRY','INCOMPATIBLE_SIZE_OR_SUPPORT'),
            visibility='PARTIAL_OR_UNCERTAIN',uncertainty_m=.03)
        self.masks={self.anchor.mask_refs[0]:self.full_mask,ref:fragment}

    def associate(self,anchors=None,partial=None):
        return self.est._associate_current_fragments(
            (anchors or [self.anchor])+[partial or self.fragment],self.masks,self.packet)

    def test_clipped_fit_does_not_invent_another_carton(self):
        original={ref:mask.copy() for ref,mask in self.masks.items()}
        result=self.associate()
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].position_m,self.anchor.position_m)
        self.assertEqual(result[0].posture,'UPRIGHT')
        self.assertFalse(result[0].failures)
        self.assertEqual(len(result[0].mask_refs),2)
        self.assertEqual(result[0].quality['current_partial_views_associated'],1)
        for ref,mask in original.items():np.testing.assert_array_equal(mask,self.masks[ref])

    def test_partial_without_current_complete_anchor_remains_unknown(self):
        result=self.est._associate_current_fragments([self.fragment],self.masks,self.packet)
        self.assertEqual(result,[self.fragment])

    def test_foreground_outliers_are_not_cropped_to_force_association(self):
        frame=self.packet.cameras['left_wrist_camera'];xyz=deproject(frame)
        xyz[self.masks[self.fragment.mask_refs[0]]]+=np.array([.1,0,0])
        with patch('workstation.perception.rgbd_cartons.deproject',return_value=xyz):
            self.assertEqual(len(self.associate()),2)

    def test_two_distinct_compatible_cuboids_keep_identity_ambiguous(self):
        other=replace(self.anchor,position_m=(.012,-.1,.7825),track_id='other')
        ref=self.fragment.mask_refs[0];mask=self.full_mask.copy()
        mask[:,:158]=False;mask[:,168:]=False;self.masks[ref]=mask
        self.assertEqual(len(self.associate([self.anchor,other])),3)

    def test_agreeing_current_complete_views_do_not_create_false_ambiguity(self):
        other=replace(self.anchor,cameras=('right_wrist_camera',),position_m=(.001,-.1,.7825))
        self.assertEqual(len(self.associate([self.anchor,other])),2)

    def test_same_camera_detection_is_not_silently_absorbed(self):
        partial=replace(self.fragment,cameras=('scene_camera',))
        self.assertEqual(len(self.associate(partial=partial)),2)

    def test_exposure_mismatch_or_missing_partial_view_is_not_attached(self):
        for time,status in ((.9,'OK'),(1.,'MISSING')):
            cameras=dict(self.packet.cameras);statuses=dict(self.packet.camera_status)
            frame=cameras['left_wrist_camera']
            states={name:replace(state,sample_time_s=time) for name,state in frame.robot_state_at_frame.items()}
            cameras['left_wrist_camera']=replace(frame,sample_time_s=time,robot_state_at_frame=states)
            if status!='OK':cameras['left_wrist_camera']=None
            statuses['left_wrist_camera']=status
            old=self.packet;self.packet=replace(old,cameras=cameras,camera_status=statuses)
            self.assertEqual(len(self.associate()),2);self.packet=old

    def test_tiny_or_invalid_depth_fragment_is_not_enough(self):
        ref=self.fragment.mask_refs[0];mask=self.masks[ref].copy()
        self.masks[ref][:]=False;self.masks[ref][mask.nonzero()[0][:10],mask.nonzero()[1][:10]]=True
        self.assertEqual(len(self.associate()),2)
        self.masks[ref]=mask
        cameras=dict(self.packet.cameras);frame=cameras['left_wrist_camera']
        cameras['left_wrist_camera']=replace(frame,valid_depth=np.zeros(frame.depth_m.shape,bool))
        self.packet=replace(self.packet,cameras=cameras)
        self.assertEqual(len(self.associate()),2)


if __name__=='__main__':unittest.main()
