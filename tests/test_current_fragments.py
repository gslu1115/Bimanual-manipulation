"""Current RGB-D surface association, independent of clipped fitted centres."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from tests.test_scene_estimate import packet
from workstation.perception.rgbd_cartons import CartonEstimator,deproject
from workstation.perception.scene_estimate import ScenePriors


class CurrentFragmentTests(unittest.TestCase):
    def test_image_connected_tops_at_two_depths_preserve_every_return(self):
        from workstation.perception.rgbd_cartons import split_separate_tops
        yy,xx=np.indices((20,30));mask=np.ones(xx.shape,bool)
        xyz=np.stack((xx*.002,yy*.002,np.where(xx<15,.815,.855)),axis=-1)
        parts=split_separate_tops(mask,mask,xyz,.008)
        self.assertEqual(sorted(int(p.sum()) for p in parts),[300,300])
        np.testing.assert_array_equal(parts[0]^parts[1],mask)
        xyz[:,:,2]=.815
        self.assertEqual(len(split_separate_tops(mask,mask,xyz,.008)),1)

    def test_neighbouring_coplanar_surfaces_require_a_measured_lateral_gap(self):
        from workstation.perception.rgbd_cartons import split_separated_depths
        mask=np.ones((12,12),bool)
        xyz=np.zeros((12,12,3))
        xyz[:,:,1]=np.arange(12)[:,None]*.004
        xyz[6:,:,1]+=.22
        parts=split_separated_depths(mask,mask,xyz,.23,lateral_gap=.006)
        self.assertEqual([part.sum() for part in parts],[72,72])
        np.testing.assert_array_equal(parts[0]^parts[1],mask)
        self.assertEqual(len(split_separated_depths(mask,mask,xyz,.28,lateral_gap=.006)),1)

    def test_one_image_component_can_contain_two_physically_separated_depth_groups(self):
        from workstation.perception.rgbd_cartons import split_separated_depths
        xyz=np.zeros((12,12,3));mask=np.ones((12,12),bool)
        xyz[:6,:,0]=.15;xyz[6:,:,0]=-.06;xyz[:6,:,2]=.78;xyz[6:,:,2]=.95
        parts=split_separated_depths(mask,mask,xyz,.16)
        self.assertEqual([part.sum() for part in parts],[72,72])
        np.testing.assert_array_equal(parts[0]|parts[1],mask)
        self.assertFalse(np.any(parts[0]&parts[1]))
        self.assertEqual(len(split_separated_depths(mask,mask,xyz,.3)),1)
        xyz[6:,:,2]=.79
        self.assertEqual(len(split_separated_depths(mask,mask,xyz,.16)),1)

    def test_total_span_can_disprove_one_box_with_a_sub_diagonal_depth_gap(self):
        from workstation.perception.rgbd_cartons import split_separated_depths
        mask=np.ones((12,12),bool);xyz=np.zeros((12,12,3))
        xyz[:6,:,2]=np.linspace(.77,.81,6)[:,None]
        xyz[6:,:,2]=np.linspace(.94,.99,6)[:,None]
        parts=split_separated_depths(mask,mask,xyz,.16)
        self.assertEqual([part.sum() for part in parts],[72,72])
        np.testing.assert_array_equal(parts[0]|parts[1],mask)
        self.assertEqual(len(split_separated_depths(mask,mask,xyz,.24)),1)

    def test_two_stationary_face_bounds_retain_shared_depth_and_reject_unexplained_points(self):
        from types import SimpleNamespace
        from workstation.perception.stationary_surface import bound_neighbour_faces
        mask=np.ones((18,41),bool);xyz=np.zeros((*mask.shape,3))
        xyz[:,:,0]=np.linspace(-.099,.099,41);xyz[:,:,1]=-.03
        xyz[:,:,2]=np.linspace(.761,.808,18)[:,None]
        packet=SimpleNamespace(cameras={'wrist':SimpleNamespace(sample_time_s=1.,valid_depth=mask)},
            camera_status={'wrist':'OK'},observation_time_s=1.)
        original=replace(self.anchor,failures=('PARTIAL_OR_MERGED_GEOMETRY',),mask_refs=('wrist/1/a',))
        scene=replace(self.est.estimate(self.packet),objects=(original,),masks={'wrist/1/a':mask})
        refs={key:dict(position_m=[x,0.,.785],size_m=[.1,.06,.05],axis_yaw_rad=0.)
              for key,x in [('left',-.052),('right',.052)]}
        with patch('workstation.perception.stationary_surface.deproject',return_value=xyz):
            result=bound_neighbour_faces(packet,scene,refs)
            self.assertEqual(len(result.objects),2)
            first,second=[result.masks[o.mask_refs[0]] for o in result.objects]
            np.testing.assert_array_equal(first|second,mask)
            self.assertTrue(np.any(first&second))
            xyz[0,0,0]=.3
            self.assertIs(bound_neighbour_faces(packet,scene,refs).objects[0],original)

    def test_clipped_neighbour_side_bounds_collision_only_from_unique_current_points(self):
        from types import SimpleNamespace
        from workstation.perception.stationary_surface import bound_neighbour_faces
        mask=np.ones((12,12),bool)
        xyz=np.zeros((12,12,3))
        xyz[:,:,0]=np.linspace(-.04,.04,12)
        xyz[:,:,1]=-.03
        xyz[:,:,2]=np.linspace(.76,.80,12)[:,None]
        frame=SimpleNamespace(sample_time_s=1.,valid_depth=mask)
        packet=SimpleNamespace(cameras={'wrist':frame},camera_status={'wrist':'OK'},observation_time_s=1.)
        ref=dict(position_m=[0.,0.,.785],size_m=[.1,.06,.05],axis_yaw_rad=0.)
        original=replace(self.anchor,failures=('PARTIAL_OR_MERGED_GEOMETRY',),mask_refs=('wrist/1/a',))
        scene=replace(self.est.estimate(self.packet),objects=(original,),masks={'wrist/1/a':mask})
        with patch('workstation.perception.stationary_surface.deproject',return_value=xyz):
            result=bound_neighbour_faces(packet,scene,{'a':ref}).objects[0]
            self.assertEqual(result.visibility,'STATIONARY_COLLISION_ENVELOPE')
            self.assertEqual(result.posture,'UNKNOWN')
            self.assertEqual(result.failures,original.failures)
            self.assertTrue(np.all(np.abs(np.asarray(ref['position_m'])-result.position_m)+
                np.asarray(ref['size_m'])/2<=np.asarray(result.size_m)/2))
            self.assertIs(bound_neighbour_faces(packet,scene,{'a':ref,'b':ref}).objects[0],original)
            frame.sample_time_s=.9
            self.assertIs(bound_neighbour_faces(packet,scene,{'a':ref}).objects[0],original)

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
        frame=self.packet.cameras['left_wrist_camera'];xyz=deproject(frame).copy()
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
