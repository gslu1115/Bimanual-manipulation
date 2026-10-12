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

    def test_current_planes_join_ids_with_measured_pose_but_no_semantic_claim(self):
        result=self.associate()
        self.assertEqual(len(result.objects),1)
        obj=result.objects[0]
        self.assertEqual(obj.track_id,self.anchor.track_id)
        np.testing.assert_allclose(obj.position_m,CENTRE,atol=.002)
        self.assertEqual(obj.posture,'UNKNOWN');self.assertFalse(obj.failures)
        self.assertEqual(obj.visibility,'OBSERVED_CUBOID')
        self.assertIs(result.masks,self.scene.masks)
        self.assertEqual(set(obj.mask_refs),set(self.scene.masks))
        self.assertEqual(self.est._anchors[self.anchor.track_id].time_s,self.packet.observation_time_s)
        np.testing.assert_allclose(self.est._anchors[self.anchor.track_id].position_m,
                                   obj.measured_cuboid['position_m'])
        self.assertEqual(self.anchor.time_s,.925)
        self.assertEqual(obj.measured_cuboid['time_s'],self.packet.observation_time_s)
        np.testing.assert_allclose(obj.measured_cuboid['position_m'],CENTRE,atol=.002)

    def test_collision_bound_uses_measured_cuboid_without_asserting_upright(self):
        from workstation.planning.collision import scene_obstacles,sphere_box_clearance
        result=self.associate();obj=result.objects[0]
        old=scene_obstacles(self.scene,None,extra_margin=0.)[0]
        box=scene_obstacles(result,None,extra_margin=0.)[0]
        self.assertLess(box.size[2],old.size[2])
        np.testing.assert_allclose(box.centre,CENTRE,atol=.002)
        # A probe actually inside the carton stays colliding; a probe above
        # its measured top is not blocked by the unrelated inflated proxy.
        self.assertLess(sphere_box_clearance(CENTRE,.01,box),0.)
        self.assertGreater(sphere_box_clearance(CENTRE+[0,0,.045],.01,box),.002)
        self.assertEqual(obj.posture,'UNKNOWN');self.assertFalse(obj.failures)

    def test_missing_or_stale_proof_retains_conservative_collision_bound(self):
        from workstation.planning.collision import scene_obstacles
        result=self.associate();obj=result.objects[0];proof=obj.measured_cuboid
        for invalid in (dict(proof,time_s=.9),dict(proof,coordinate_constraints=[]),
                        dict(proof,mask_refs=proof['mask_refs'][:-1]),dict(proof,size_m=[.05,.055,.045])):
            with self.subTest(invalid=invalid):
                scene=replace(result,objects=(replace(obj,measured_cuboid=invalid),))
                self.assertGreater(scene_obstacles(scene,None)[0].size[2],.1)
        old=replace(result,assembled_time_s=1.2)
        self.assertGreater(scene_obstacles(old,None)[0].size[2],.1)

    def test_single_partial_instance_also_gets_only_current_measured_bounds(self):
        original=replace(self.scene.objects[0],mask_refs=tuple(self.scene.masks))
        scene=replace(self.scene,objects=(original,))
        result=self.associate(scene=scene)
        self.assertEqual(len(result.objects),1)
        self.assertIsNotNone(result.objects[0].measured_cuboid)
        self.assertEqual(result.objects[0].posture,'UNKNOWN')

    def with_current_tape(self):
        cameras={}
        for name,frame in self.packet.cameras.items():
            if frame is None:cameras[name]=None;continue
            xyz=deproject(frame);rgb=frame.rgb.copy()
            stripe=(np.abs(xyz[...,2]-(CENTRE[2]+PRIORS.carton_size_m[2]/2))<.001)&(
                    np.abs(xyz[...,1]-CENTRE[1])<.006)
            rgb[stripe]=[205,160,100]
            cameras[name]=replace(frame,rgb=rgb)
        return replace(self.packet,cameras=cameras)

    def test_current_positive_rgb_resolves_semantics_on_independent_geometry(self):
        from workstation.perception.scene_estimate import pose_is_observed
        current=self.with_current_tape();result=self.associate(packet=current);obj=result.objects[0]
        self.assertEqual(obj.posture,'UPRIGHT');self.assertFalse(obj.failures)
        self.assertEqual(obj.visibility,'OBSERVED_CUBOID')
        self.assertTrue(pose_is_observed(obj,result))
        self.assertGreaterEqual(obj.quality['tape_camera_count'],1)
        bad=replace(obj,position_m=tuple(np.asarray(obj.position_m)+[.01,0,0]))
        self.assertFalse(pose_is_observed(bad,replace(result,objects=(bad,))))
        self.assertFalse(pose_is_observed(obj,replace(result,assembled_time_s=1.2)))

    def test_observed_pose_does_not_turn_partial_top_coverage_into_full_visibility(self):
        current=self.with_current_tape();masks={}
        for ref,mask in self.scene.masks.items():
            xyz=deproject(current.cameras[ref.split('/')[0]])
            side=np.abs(xyz[...,2]-(CENTRE[2]+PRIORS.carton_size_m[2]/2))>.003
            masks[ref]=mask&(side|(xyz[...,0]<CENTRE[0]+.020))
        result=self.associate(packet=current,scene=replace(self.scene,masks=masks));obj=result.objects[0]
        self.assertEqual(obj.posture,'UPRIGHT');self.assertEqual(obj.visibility,'OBSERVED_CUBOID')
        self.assertGreater(obj.quality['footprint_coverage'],.6)
        self.assertLess(obj.quality['footprint_coverage'],.8)
        self.assertGreater(obj.quality['footprint_error_m'],.020)
        self.assertGreaterEqual(obj.measured_cuboid['surface_inlier_ratio'],.95)

    def test_measured_semantic_pose_can_initialize_current_holding_tracker(self):
        from workstation.perception.held_surface import CuboidHoldingTracker,HeldSurfaceTracker
        current=self.with_current_tape();result=self.associate(packet=current);obj=result.objects[0]
        tcp=CENTRE+[0,0,.009];rotation=np.diag([1.,-1.,-1.])
        tracker=CuboidHoldingTracker(obj,tcp,rotation,PRIORS.carton_size_m,PRIORS)
        proof=tracker.observe(current,result,tcp,rotation)
        self.assertIsNotNone(proof);self.assertTrue(proof['geometric_pose_measured'])
        upright=HeldSurfaceTracker(obj,tcp,rotation,PRIORS.carton_size_m)
        self.assertIsNotNone(upright.observe(current,result,tcp,rotation))

    def test_model_refresh_accepts_measured_pose_without_recentering_grasp(self):
        from workstation.planning.graspgen_candidates import candidates_from_result,revalidate_candidate
        current=self.with_current_tape();scene=self.associate(packet=current);obj=scene.objects[0]
        old=replace(obj,time_s=.925)
        pose=np.eye(4);pose[:3,:3]=np.diag([1.,-1.,-1.]);pose[:3,3]=CENTRE+[0,0,.009]
        response=dict(base_poses_workcell=[pose],scores=[.8],request_id='same',observation_time_s=.925,
                      metadata={'gripper_name':'franka_panda'})
        calibration={'verified':True,'T_graspgen_base_from_project_tcp':np.eye(4).tolist()}
        candidate=candidates_from_result(response,old,current.robot_state,calibration,{'scene_camera':99})[0]
        refreshed=revalidate_candidate(candidate,obj,scene,{'scene_camera':100})
        self.assertEqual(refreshed.position_m,candidate.position_m)
        self.assertEqual(refreshed.orientation_wxyz,candidate.orientation_wxyz)
        self.assertEqual(refreshed.observation_time_s,.925)
        self.assertEqual(refreshed.validated_time_s,1.)

    def test_current_packaging_conflict_is_not_replaced_by_previous_upright(self):
        from unittest.mock import patch
        with patch.object(self.est,'_flat_packaging',return_value=('UNKNOWN',{'packaging_conflict':1.},'current conflict')):
            result=self.associate(packet=self.with_current_tape())
        obj=result.objects[0]
        self.assertEqual(obj.posture,'UNKNOWN')
        self.assertIn('PACKAGING_CUE_CONFLICT',obj.failures)

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
