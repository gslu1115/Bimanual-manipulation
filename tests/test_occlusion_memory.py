import unittest
from types import SimpleNamespace
import numpy as np
from workstation.perception.scene_estimate import ScenePriors
from workstation.planning.robot_geometry import ConvexLink
from workstation.planning.observed_workspace import ObservedWorkspace
from workstation.planning.occlusion_memory import OcclusionMemory
from workstation.planning.collision import AABB


class OcclusionMemoryTests(unittest.TestCase):
    def workspace(self,time=1.,robot=False,depth=1.,valid=True,camera_x=0.):
        t=np.diag([1.,-1.,-1.,1.]);t[:3,3]=[camera_x,0.,1.8]
        image=np.full((50,50),depth,np.float32)
        if robot:image[17:34,17:34]=.6
        frame=SimpleNamespace(sample_time_s=time,robot_state_at_frame={},
            depth_m=image,valid_depth=np.full((50,50),valid,bool),
            T_workcell_from_camera_cv=t,K=np.array([[50.,0.,25.],[0.,50.,25.],[0.,0.,1.]]))
        packet=SimpleNamespace(cameras={'camera':frame},camera_status={'camera':'OK'},
            observation_time_s=time,assembled_time_s=time)
        half=np.array([.06,.06,.03])
        planes=np.c_[np.r_[np.eye(3),-np.eye(3)],-np.r_[half,half]]
        model=ConvexLink(planes,-half,half,'fixture')
        links=((model,np.array([0.,0.,1.17]),np.eye(3)),) if robot else ()
        priors=ScenePriors((.04,.02,.02),.8,(-.05,.05,-.05,.05))
        return ObservedWorkspace(packet,priors,SimpleNamespace(objects=()),robot_links=links)

    def test_measured_history_can_survive_robot_self_occlusion(self):
        old=self.workspace();current=self.workspace(2.,robot=True)
        memory=OcclusionMemory();memory.add(old)
        point=np.array([[0.,0.,1.1]])
        self.assertFalse(current._ray_free(point,next(current._frames()))[0])
        self.assertTrue(memory.free(point,current)[0])
        self.assertEqual(memory.report()['oldest_reused_age_s'],1.)

    def test_unseen_space_is_never_created_by_robot_mask(self):
        memory=OcclusionMemory();memory.add(self.workspace(depth=.65))
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),self.workspace(2.,robot=True))[0])

    def test_old_robot_volume_is_not_free_history(self):
        memory=OcclusionMemory();memory.add(self.workspace(robot=True))
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),self.workspace(2.,robot=True))[0])

    def test_expired_or_invalid_frames_cannot_clear_shadow(self):
        for old in (self.workspace(),self.workspace(valid=False)):
            memory=OcclusionMemory(max_age_s=.5);memory.add(old)
            self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),self.workspace(2.,robot=True))[0])

    def test_nonrobot_occluder_in_other_camera_vetoes_history(self):
        memory=OcclusionMemory();memory.add(self.workspace())
        current=self.workspace(2.,robot=True)
        other=self.workspace(2.,depth=.55)
        current.packet.cameras['obstacle_camera']=other.packet.cameras['camera']
        current.packet.camera_status['obstacle_camera']='OK'
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),current)[0])

    def test_visual_motion_envelope_vetoes_history(self):
        memory=OcclusionMemory();memory.add(self.workspace())
        current=self.workspace(2.,robot=True)
        current.scene.objects=(SimpleNamespace(position_m=(0.,0.,1.08),axis_yaw_rad=0.,
            size_m=(.04,.02,.02),quality={},uncertainty_m=.003,velocity_m_s=(0.,0.,.02)),)
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),current)[0])

    def test_original_image_time_transform_is_used(self):
        memory=OcclusionMemory();memory.add(self.workspace(camera_x=10.))
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),self.workspace(2.,robot=True))[0])

    def test_memory_is_bounded_and_rejects_stale_at_capture(self):
        memory=OcclusionMemory(max_age_s=1.,interval_s=.1,max_snapshots=2)
        stale=self.workspace();stale.packet.assembled_time_s=2.
        memory.add(stale);self.assertFalse(memory.snapshots)
        for time in (2.,2.2,2.4,4.):memory.add(self.workspace(time))
        self.assertEqual(len(memory.snapshots),1)

    def test_fine_query_recovers_previously_seen_shadow_not_occupancy(self):
        memory=OcclusionMemory();memory.add(self.workspace())
        current=self.workspace(2.,robot=True);current.occlusion_memory=memory
        self.assertIsNone(current.check_sphere((0.,0.,1.1),.005))
        current.occupied[:]=True
        self.assertEqual(current.check_sphere((0.,0.,1.1),.005),'OBSERVED_PATH_OBSTACLE')

    def payload(self,workspace,target_id='visual_held',centre=(0.,0.,1.13),size=(.1,.1,.04)):
        ref='camera/observed/payload';mask=np.zeros((50,50),bool);mask[22:29,22:29]=True
        workspace.scene.masks={ref:mask}
        evidence=dict(time_s=workspace.packet.observation_time_s,mask_refs=[ref],surface_inlier_ratio=1.,
            top_coverage=.5,side_points=25,plane_residual_m=.0001,surface_tilt_deg=0.)
        self.assertTrue(workspace.confirm_payload(target_id,centre,np.eye(3),size,evidence))
        return evidence

    def test_action_anchor_requires_current_payload_support_and_preserves_original_rays(self):
        memory=OcclusionMemory();memory.anchor(self.workspace())
        current=self.workspace(16.,robot=True)
        current.scene.objects=(SimpleNamespace(track_id='visual_held',mask_refs=(),position_m=(0.,0.,1.08),axis_yaw_rad=0.,
            size_m=(.04,.02,.02),quality={},uncertainty_m=.003,velocity_m_s=(0.,0.,.02)),)
        point=np.array([[0.,0.,1.1]])
        self.assertFalse(memory.free(point,current)[0])
        self.payload(current)
        self.assertTrue(memory.free(point,current)[0])
        self.assertEqual(memory.report()['anchor_reused_sample_evaluations'],1)
        self.assertEqual(memory.report()['oldest_reused_age_s'],15.)
        current.packet.assembled_time_s+=.2
        self.assertFalse(memory.free(point,current)[0])

    def test_payload_shadow_needs_supported_instance_pixels_not_only_model(self):
        memory=OcclusionMemory();memory.anchor(self.workspace())
        current=self.workspace(16.,depth=.65);self.payload(current)
        point=np.array([[0.,0.,1.1]])
        self.assertTrue(memory.free(point,current)[0])
        current.scene.masks={ref:np.zeros_like(mask) for ref,mask in current.scene.masks.items()}
        del current._payload_depth_masks
        self.assertFalse(memory.free(point,current)[0])

    def test_payload_anchor_never_fabricates_unseen_or_expired_space(self):
        for old,now in ((self.workspace(depth=.65),16.),(self.workspace(),32.)):
            memory=OcclusionMemory();memory.anchor(old)
            current=self.workspace(now,robot=True);self.payload(current)
            self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),current)[0])

    def test_other_visual_object_and_unidentified_foreground_still_veto_payload_anchor(self):
        memory=OcclusionMemory();memory.anchor(self.workspace())
        current=self.workspace(16.,robot=True);self.payload(current)
        current.scene.objects=(SimpleNamespace(track_id='other_visual',position_m=(0.,0.,1.08),axis_yaw_rad=0.,
            size_m=(.04,.02,.02),quality={},uncertainty_m=.003,velocity_m_s=(0.,0.,0.)),)
        point=np.array([[0.,0.,1.1]])
        self.assertFalse(memory.free(point,current)[0])
        current.scene.objects=()
        other=self.workspace(16.,depth=.55)
        current.packet.cameras['obstacle']=other.packet.cameras['camera'];current.packet.camera_status['obstacle']='OK'
        self.assertFalse(memory.free(point,current)[0])

    def test_missing_or_stale_surface_evidence_revokes_payload_history(self):
        current=self.workspace(16.,robot=True);evidence=self.payload(current)
        for change in (dict(time_s=15.),dict(surface_inlier_ratio=.5),dict(mask_refs=['missing/ref']),
                       dict(side_points=0),dict(surface_inlier_ratio=float('nan'))):
            invalid=dict(evidence);invalid.update(change)
            self.assertFalse(current.confirm_payload('visual_held',(0.,0.,1.13),np.eye(3),(.1,.1,.04),invalid))
            self.assertIsNone(current.payload_observation)

    def test_action_anchors_are_bounded_and_invalid_capture_is_not_pinned(self):
        memory=OcclusionMemory(max_anchors=2);stale=self.workspace();stale.packet.assembled_time_s=3.
        memory.anchor(stale);self.assertFalse(memory.anchors)
        for t in (2.,3.,4.):memory.anchor(self.workspace(t))
        self.assertEqual([w.packet.observation_time_s for w in memory.anchors],[3.,4.])
        memory.add(self.workspace(35.));self.assertFalse(memory.anchors)

    def test_current_associated_fragments_are_filtered_without_inventing_hold_support(self):
        memory=OcclusionMemory();memory.anchor(self.workspace())
        current=self.workspace(16.,depth=.65);evidence=self.payload(current)
        support=evidence['mask_refs'][0]
        current.scene.masks[support][:]=False
        fragment='camera/observed/fragment';mask=np.zeros((50,50),bool);mask[22:29,22:29]=True
        current.scene.masks[fragment]=mask
        current.scene.objects=(SimpleNamespace(track_id='visual_held',mask_refs=(fragment,)),)
        self.assertTrue(current.confirm_payload('visual_held',(0.,0.,1.13),np.eye(3),(.1,.1,.04),evidence))
        self.assertEqual(current.payload_observation['support_mask_refs'],(support,))
        self.assertEqual(current.payload_observation['mask_refs'],(support,fragment))
        self.assertTrue(memory.free(np.array([[0.,0.,1.1]]),current)[0])
        # Same instance label cannot explain a foreground far from the visual
        # payload surface. No broad mask or bounding-box clearing is allowed.
        current.packet.cameras['camera'].depth_m[:]=.55
        del current._payload_depth_masks
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),current)[0])

    def test_mixed_robot_payload_footprint_requires_union_before_erosion(self):
        import cv2
        memory=OcclusionMemory();memory.anchor(self.workspace())
        current=self.workspace(16.,depth=.65);self.payload(current)
        robot=np.zeros((50,50),bool);robot[:,:25]=True
        payload=~robot
        def mask(values,erode):
            return cv2.erode(values.astype(np.uint8),np.ones((3,3),np.uint8))!=0 if erode else values
        memory._robot_depth_mask=lambda ws,f,erode=True:mask(robot,erode)
        memory._payload_depth_mask=lambda ws,f,erode=True:mask(payload,erode)
        point=np.array([[0.,0.,1.1]])
        self.assertFalse(mask(robot,True)[25,25]);self.assertFalse(mask(payload,True)[25,25])
        self.assertTrue(memory.free(point,current)[0])
        payload[25,25]=False # One unexplained pixel must still veto history.
        self.assertFalse(memory.free(point,current)[0])

    def test_tiny_enclosed_mask_hole_needs_measured_payload_surface(self):
        memory=OcclusionMemory();memory.anchor(self.workspace())
        current=self.workspace(16.,depth=.65);self.payload(current)
        ref=next(iter(current.scene.masks));current.scene.masks[ref][25,25]=False
        original=current.scene.masks[ref].copy();point=np.array([[0.,0.,1.1]])
        self.assertTrue(memory.free(point,current)[0])
        np.testing.assert_array_equal(current.scene.masks[ref],original)
        # A foreign foreground in the hole is not a missing carton pixel.
        current.packet.cameras['camera'].depth_m[25,25]=.55
        current._depth_footprints.clear();del current._payload_depth_masks
        self.assertFalse(memory.free(point,current)[0])

    def test_large_or_exterior_segmentation_gap_is_not_repaired(self):
        memory=OcclusionMemory();memory.anchor(self.workspace());point=np.array([[0.,0.,1.1]])
        for exterior in (False,True):
            current=self.workspace(16.,depth=.65);self.payload(current);ref=next(iter(current.scene.masks))
            if exterior:current.scene.masks[ref][24:29,24:29]=False
            else:current.scene.masks[ref][23:27,23:27]=False
            self.assertFalse(memory.free(point,current)[0])

    def test_single_pixel_dropout_on_occlusion_edge_requires_current_surface_depth(self):
        memory=OcclusionMemory();memory.anchor(self.workspace())
        current=self.workspace(16.,depth=.65);self.payload(current);ref=next(iter(current.scene.masks))
        current.scene.masks[ref][22:26,25]=False # Slit connected to outside mask.
        original=current.scene.masks[ref].copy();point=np.array([[0.,0.,1.1]])
        self.assertTrue(memory.free(point,current)[0])
        np.testing.assert_array_equal(current.scene.masks[ref],original)
        current.packet.cameras['camera'].depth_m[25,25]=.55
        current._depth_footprints.clear();del current._payload_depth_masks
        self.assertFalse(memory.free(point,current)[0])

    def test_robot_background_boundary_uses_each_pixels_actual_depth(self):
        memory=OcclusionMemory();memory.add(self.workspace())
        current=self.workspace(2.,robot=True);frame=next(current._frames())
        point=np.array([[0.,0.,1.1]])
        # Query optical z=.7. The left footprint column is known background
        # at z=1, while other pixels are supported robot at z=.6.
        frame.depth_m[24:27,24]=1.
        self.assertFalse(current._ray_free(point,frame)[0])
        self.assertTrue(memory.free(point,current)[0])
        # Unidentified FOREGROUND at that same pixel must still veto history.
        frame.depth_m[25,24]=.55
        current._depth_footprints.clear();current._robot_depth_masks.clear()
        self.assertFalse(memory.free(point,current)[0])

    def test_mixed_self_background_still_requires_valid_depth_and_old_ray(self):
        point=np.array([[0.,0.,1.1]])
        for invalid_depth in (False,True):
            memory=OcclusionMemory();memory.add(self.workspace(depth=.65) if not invalid_depth else self.workspace())
            current=self.workspace(2.,robot=True);frame=next(current._frames())
            frame.depth_m[24:27,24]=1.
            if invalid_depth:frame.valid_depth[25,24]=False
            self.assertFalse(memory.free(point,current)[0])

    def near_fixed_surface(self):
        current=self.workspace(2.,robot=True)
        frame=next(current._frames());frame.depth_m[24:27,24]=.701
        current.static=(AABB('fixed_support',(0.,0.,1.049),(.2,.2,.1)),)
        return current,frame

    def test_measured_near_fixed_surface_does_not_veto_an_independent_old_free_ray(self):
        memory=OcclusionMemory();memory.add(self.workspace())
        current,frame=self.near_fixed_surface();point=np.array([[0.,0.,1.1]])
        self.assertFalse(current._ray_free(point,frame)[0])
        self.assertTrue(memory.free(point,current)[0])

    def test_fixed_surface_never_creates_free_history_or_explains_new_foreground(self):
        point=np.array([[0.,0.,1.1]])
        memory=OcclusionMemory();memory.add(self.workspace(depth=.65))
        current,_=self.near_fixed_surface()
        self.assertFalse(memory.free(point,current)[0])
        memory=OcclusionMemory();memory.add(self.workspace())
        current,frame=self.near_fixed_surface();frame.depth_m[25,24]=.693
        self.assertFalse(memory.free(point,current)[0])

    def test_fixed_surface_does_not_explain_behind_surface_or_invalid_depth(self):
        memory=OcclusionMemory();memory.add(self.workspace())
        current,frame=self.near_fixed_surface()
        self.assertFalse(memory.free(np.array([[0.,0.,1.096]]),current)[0])
        current,frame=self.near_fixed_surface();frame.valid_depth[25,24]=False
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),current)[0])

    def two_view_payload(self):
        current=self.workspace(16.,depth=.65);evidence=self.payload(current)
        original=next(iter(current.scene.masks));current.scene.masks[original][:]=False
        other=self.workspace(16.,depth=.65)
        current.packet.cameras['other']=other.packet.cameras['camera'];current.packet.camera_status['other']='OK'
        ref='other/observed/payload';mask=np.zeros((50,50),bool);mask[22:29,22:29]=True
        current.scene.masks[ref]=mask;evidence['mask_refs']=[ref]
        self.assertTrue(current.confirm_payload('visual_held',(0.,0.,1.13),np.eye(3),(.1,.1,.04),evidence))
        return current,original,evidence

    def test_current_other_view_can_explain_unlabelled_metric_payload_surface(self):
        memory=OcclusionMemory();memory.anchor(self.workspace())
        current,ref,_=self.two_view_payload();original=current.scene.masks[ref].copy()
        self.assertTrue(memory.free(np.array([[0.,0.,1.1]]),current)[0])
        np.testing.assert_array_equal(current.scene.masks[ref],original)

    def test_cross_view_payload_filter_keeps_foreign_depth_and_invalid_pixels_blocked(self):
        for invalid in (False,True):
            memory=OcclusionMemory();memory.anchor(self.workspace());current,_,_=self.two_view_payload()
            frame=current.packet.cameras['camera']
            if invalid:
                # One invalid view must not suppress another valid view's
                # evidence. Invalidate BOTH footprints to test no clearance.
                frame.valid_depth[25,25]=False
                current.packet.cameras['other'].valid_depth[25,25]=False
            else:frame.depth_m[25,25]=.643 # 7 mm beyond supported top, over3 mm gate.
            self.assertFalse(memory._payload_depth_mask(current,frame,erode=False)[25,25])
            self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),current)[0])

    def test_cross_view_payload_filter_cannot_use_stale_other_view_or_unseen_history(self):
        current,_,evidence=self.two_view_payload();current.packet.cameras['other'].sample_time_s=15.
        self.assertFalse(current.confirm_payload('visual_held',(0.,0.,1.13),np.eye(3),(.1,.1,.04),evidence))
        memory=OcclusionMemory();memory.anchor(self.workspace())
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),current)[0])
        current,_,_=self.two_view_payload();memory=OcclusionMemory();memory.anchor(self.workspace(depth=.65))
        self.assertFalse(memory.free(np.array([[0.,0.,1.1]]),current)[0])


if __name__=='__main__':unittest.main()
