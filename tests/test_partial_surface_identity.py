"""Clipped views may share an identity while retaining uncertain geometry."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from tests import test_current_fragments as fixtures
from workstation.perception.rgbd_cartons import deproject
from workstation.perception.scene_estimate import pose_is_observed


class PartialSurfaceIdentityTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.CurrentFragmentTests();fixture.setUp()
        self.est,self.packet,self.masks=fixture.est,fixture.packet,fixture.masks
        self.complete=fixture.anchor
        self.first=replace(fixture.anchor,posture='UNKNOWN',visibility='PARTIAL_OR_UNCERTAIN',
            failures=('PARTIAL_OR_MERGED_GEOMETRY',),position_m=(-.004,-.1,.7825))
        self.second=replace(fixture.fragment,position_m=(-.004,-.1,.755))
        # Shared observed pixels, but one clipped fit wrongly selects a tall face.
        self.masks[self.second.mask_refs[0]]=fixture.full_mask.copy()
        self.est._anchors={}

    def associate(self,objects=None):
        return self.est._associate_overlapping_partial_views(
            objects or [self.first,self.second],self.masks,self.packet)

    def test_shared_current_surface_joins_identity_without_inventing_pose(self):
        masks={k:v.copy() for k,v in self.masks.items()}
        result=self.associate();self.assertEqual(len(result),1)
        obj=result[0]
        self.assertEqual(obj.posture,'UNKNOWN');self.assertTrue(obj.failures)
        self.assertEqual(obj.visibility,'PARTIAL_OR_UNCERTAIN')
        self.assertIsNone(obj.measured_cuboid);self.assertIsNone(obj.velocity_m_s)
        self.assertEqual(set(obj.mask_refs),set(self.masks))
        for ref,mask in masks.items():np.testing.assert_array_equal(mask,self.masks[ref])

    def test_estimator_preserves_existing_target_id_after_two_clipped_fits(self):
        self.est._tracks={self.complete.track_id:replace(self.complete,time_s=.9)}
        self.est._last_time=.9
        detections=[([self.first],{self.first.mask_refs[0]:self.masks[self.first.mask_refs[0]]}),
                    ([self.second],{self.second.mask_refs[0]:self.masks[self.second.mask_refs[0]]})]
        with patch.object(self.est,'_detect',side_effect=detections):scene=self.est.estimate(self.packet)
        self.assertEqual(len(scene.objects),1)
        self.assertEqual(scene.objects[0].track_id,self.complete.track_id)
        self.assertFalse(pose_is_observed(scene.objects[0],scene))

    def test_actual_separate_surface_does_not_merge(self):
        original=deproject
        def moved(frame):
            xyz=original(frame).copy()
            if frame.name=='left_wrist_camera':xyz+=np.array([.08,0,0])
            return xyz
        with patch('workstation.perception.rgbd_cartons.deproject',side_effect=moved):
            self.assertEqual(len(self.associate()),2)

    def test_overlapping_view_with_second_box_cannot_be_absorbed(self):
        original=deproject
        def contaminated(frame):
            xyz=original(frame).copy()
            if frame.name=='left_wrist_camera':
                v,u=np.nonzero(self.masks[self.second.mask_refs[0]])
                xyz[v[::5],u[::5]]+=np.array([.12,0,0])
            return xyz
        with patch('workstation.perception.rgbd_cartons.deproject',side_effect=contaminated):
            self.assertEqual(len(self.associate()),2)

    def test_competing_same_camera_instances_keep_all_tracks(self):
        alias=replace(self.second,mask_refs=('left_wrist_camera/100/other',))
        self.masks[alias.mask_refs[0]]=self.masks[self.second.mask_refs[0]].copy()
        self.assertEqual(len(self.associate([self.first,self.second,alias])),3)

    def test_missing_or_old_frame_cannot_supply_identity(self):
        for key in ('stale','missing'):
            old=self.packet;cameras=dict(old.cameras);statuses=dict(old.camera_status)
            frame=cameras['left_wrist_camera']
            if key=='missing':cameras['left_wrist_camera']=None;statuses['left_wrist_camera']='MISSING'
            else:
                states={k:replace(v,sample_time_s=.9) for k,v in frame.robot_state_at_frame.items()}
                cameras['left_wrist_camera']=replace(frame,sample_time_s=.9,robot_state_at_frame=states)
            self.packet=replace(old,cameras=cameras,camera_status=statuses)
            self.assertEqual(len(self.associate()),2);self.packet=old


if __name__=='__main__':unittest.main()
