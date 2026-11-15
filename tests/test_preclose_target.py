"""Current partial RGB-D supports close without becoming a full-pose measurement."""
import unittest
from dataclasses import replace
import numpy as np

from tests.test_cuboid_pose import synthetic_observation, visual_object, CENTRE, SIZE, PRIORS, turn
from workstation.perception.rgbd_cartons import CartonEstimator, deproject
from workstation.perception.preclose_target import (PreCloseTarget, preclose_supported, reconcile_preclose_target,
                                                   reconcile_reacquired_target)
from workstation.perception.cuboid_pose import (measured_cuboid_supported,
    grasp_constrained_cuboid_supported, reference_constrained_cuboid_supported)
from workstation.perception.held_surface import CuboidHoldingTracker, HeldSurfaceTracker
from workstation.perception.scene_estimate import pose_is_observed


class PreCloseTargetTests(unittest.TestCase):
    def setUp(self):
        self.priors=replace(PRIORS,support_z_m=float(CENTRE[2]-SIZE[2]/2))
        self.packet,self.scene=self.partial()
        self.reference=replace(visual_object(self.scene,np.eye(3)),time_s=.9)
        self.reference_scene=replace(self.scene,observation_time_s=.9,assembled_time_s=.9,
                                     objects=(self.reference,))
        self.check=PreCloseTarget(self.reference,self.reference_scene,self.priors)

    def partial(self,centre=CENTRE,rotation=None,tape=True):
        p,s=synthetic_observation(np.eye(3) if rotation is None else rotation,centre,
            views=([0.,-.1001,1.35],[0.,-.1001,1.35]))
        cameras={};masks={}
        for name,frame in p.cameras.items():
            if frame is None:cameras[name]=None;continue
            xyz=deproject(frame);rgb=frame.rgb.copy()
            if tape:
                stripe=(np.abs(xyz[...,2]-(centre[2]+SIZE[2]/2))<.001)&(np.abs(xyz[...,1]-centre[1])<.006)
                rgb[stripe]=[205,160,100]
            cameras[name]=replace(frame,rgb=rgb)
            ref=name+'/100/carton'
            masks[ref]=s.masks[ref]&(xyz[...,0]>centre[0]-.035)&(xyz[...,0]<centre[0]+.035)
        return replace(p,cameras=cameras),replace(s,masks=masks)

    def test_partial_top_and_tape_keep_only_one_hidden_axis_conditional(self):
        proof=self.check.confirm(self.packet,self.scene)
        self.assertTrue(proof['supported'],proof)
        m=proof['surface']
        self.assertTrue(preclose_supported(proof,1.))
        self.assertTrue(reference_constrained_cuboid_supported(m,1.))
        self.assertFalse(measured_cuboid_supported(m,1.))
        self.assertFalse(grasp_constrained_cuboid_supported(m,1.))
        self.assertFalse(proof['full_pose_measured']);self.assertFalse(proof['grasp_proven'])
        self.assertEqual(m['predicted_translation_axes'],[0])
        self.assertEqual(proof['reference_time_s'],.9)

    def test_sparse_end_face_completes_a_two_view_preclose_pose(self):
        from workstation.perception.cuboid_pose import fit_cuboid_pose
        p,s=synthetic_observation(np.eye(3),views=([0.,-.101,1.35],[-.32,-.10,1.08]))
        masks={};cameras=dict(p.cameras)
        for ref,mask in s.masks.items():
            name=ref.split('/')[0];f=cameras[name];xyz=deproject(f)
            top=np.abs(xyz[:,:,2]-CENTRE[2]-SIZE[2]/2)<.001
            clip=top & (np.abs(xyz[:,:,0]-CENTRE[0])<.035)
            # Disconnected rows of real side returns have no normal stencil.
            if name!='scene_camera':
                clip|=(np.abs(xyz[:,:,0]-CENTRE[0]+SIZE[0]/2)<.001)&(np.indices(mask.shape)[0]%3==0)
            masks[ref]=mask & clip
            rgb=f.rgb.copy();rgb[top & (np.abs(xyz[:,:,1]-CENTRE[1])<.006)]=[205,160,100]
            cameras[name]=replace(f,rgb=rgb)
        p=replace(p,cameras=cameras);s=replace(s,masks=masks)
        def fit(scene,enabled):
            return fit_cuboid_pose(p,scene,CENTRE,np.eye(3),SIZE,PRIORS,
                                   allow_reference_translation=True,measure_side_interiors=enabled)
        self.assertFalse(measured_cuboid_supported(fit(s,False),1.))
        m=fit(s,True)
        self.assertTrue(measured_cuboid_supported(m,1.),m)
        self.assertTrue(any(c.get('extraction') for c in m['coordinate_constraints']))
        np.testing.assert_allclose(m['position_m'],CENTRE,atol=.001)
        one_view=replace(s,masks=dict([next(iter(masks.items()))]))
        self.assertFalse(measured_cuboid_supported(fit(one_view,True),1.))

    def test_no_current_direction_missing_frames_and_contamination_are_rejected(self):
        p,s=self.partial(tape=False)
        cases=[(p,s),(self.missing_packet(),self.scene)]
        masks={key:mask.copy() for key,mask in self.scene.masks.items()}
        for mask in masks.values():mask[20:50,20:50]=True
        cases.append((self.packet,replace(self.scene,masks=masks)))
        for p,s in cases:
            with self.subTest():self.assertFalse(self.check.confirm(p,s)['supported'])

    def test_carry_reacquisition_can_constrain_one_hidden_translation(self):
        from workstation.perception.held_surface import reacquire_tilted_top
        args=(self.packet,self.scene,CENTRE,np.eye(3),SIZE,self.priors)
        self.assertIsNone(reacquire_tilted_top(*args))
        measured=reacquire_tilted_top(*args,allow_grasp_translation=True)
        self.assertTrue(grasp_constrained_cuboid_supported(measured,1.))
        self.assertEqual(measured['predicted_translation_axes'],[0])
        self.assertFalse(measured['geometric_pose_measured'])

    def test_stationary_reference_accepts_measured_rectangular_top_but_not_a_short_patch(self):
        m=dict(self.check.confirm(self.packet,self.scene)['surface'],tape_direction=None,
               top_rectangle=dict(measured_xy_m=list(SIZE[:2]),hull_fraction=.92))
        self.assertTrue(reference_constrained_cuboid_supported(m,1.))
        self.assertFalse(measured_cuboid_supported(m,1.))
        m['top_rectangle']['measured_xy_m']=[SIZE[0]-.012,SIZE[1]]
        self.assertFalse(reference_constrained_cuboid_supported(m,1.))

    def test_latched_grasp_can_use_full_top_rectangle_for_orientation(self):
        from workstation.perception.cuboid_pose import fit_cuboid_pose
        m=fit_cuboid_pose(self.packet,self.scene,CENTRE,np.eye(3),SIZE,self.priors,
                          allow_grasp_translation=True)
        rectangle=dict(measured_xy_m=list(SIZE[:2]),hull_fraction=.96)
        evidence=dict(m,tape_direction=None,top_rectangle=rectangle)
        self.assertTrue(grasp_constrained_cuboid_supported(evidence,1.))
        self.assertFalse(measured_cuboid_supported(evidence,1.))
        self.assertFalse(grasp_constrained_cuboid_supported(evidence,1.2))
        self.assertFalse(grasp_constrained_cuboid_supported(
            dict(evidence,top_rectangle=dict(rectangle,hull_fraction=.6)),1.))

    def test_two_views_can_reacquire_shifted_top_but_one_view_cannot(self):
        from workstation.perception.held_surface import reacquire_tilted_top
        def reacquire(scene, shift):
            return reacquire_tilted_top(self.packet, scene, CENTRE+[0,0,shift],
                                        np.eye(3), SIZE, self.priors, allow_grasp_translation=True)
        fit = reacquire(self.scene, .011)
        self.assertIsNotNone(fit)
        self.assertTrue(grasp_constrained_cuboid_supported(fit, 1.))
        self.assertAlmostEqual(fit['normal_offset_m'], -.011, places=3)
        one_view = replace(self.scene, masks=dict([next(iter(self.scene.masks.items()))]))
        self.assertIsNone(reacquire(one_view, .011))
        self.assertIsNone(reacquire(self.scene, .014))

    def test_visible_motion_and_tilt_are_rejected(self):
        for centre,rotation in ((CENTRE+[0,.006,0],None),(CENTRE,turn(1,20))):
            p,s=self.partial(centre,rotation)
            self.assertFalse(self.check.confirm(p,s)['supported'])

    def test_reference_and_frame_expiry_are_not_retimestamped(self):
        check=PreCloseTarget(replace(self.reference,time_s=-6.),
            replace(self.reference_scene,observation_time_s=-6.,assembled_time_s=-6.),self.priors)
        self.assertEqual(check.confirm(self.packet,self.scene)['reason'],'PRECLOSE_REFERENCE_EXPIRED')
        cameras={}
        for name,f in self.packet.cameras.items():
            cameras[name]=None if f is None else replace(f,sample_time_s=.7,
                robot_state_at_frame={arm:replace(state,sample_time_s=.7) for arm,state in f.robot_state_at_frame.items()})
        self.assertFalse(self.check.confirm(replace(self.packet,cameras=cameras),self.scene)['supported'])

    def test_only_full_current_geometry_renews_stationary_reference(self):
        partial=self.check.confirm(self.packet,self.scene)
        self.assertIs(self.check.renewed(self.scene,partial),self.check)
        packet,scene=synthetic_observation(np.eye(3))
        obj=replace(self.reference,time_s=1.,mask_refs=tuple(scene.masks))
        scene=replace(scene,objects=(obj,))
        proof=self.check.confirm(packet,scene)
        reconciled=reconcile_preclose_target(CartonEstimator(self.priors),scene,proof)
        new=self.check.renewed(reconciled,proof)
        self.assertEqual(new.obj.time_s,1.)
        self.assertIs(new.original,self.check)

    def test_expired_pick_reacquires_a_misclassified_box_from_height_and_current_views(self):
        packet,scene=synthetic_observation(np.eye(3))
        old=replace(self.reference,time_s=-8.)
        reference=PreCloseTarget(old,replace(self.reference_scene,observation_time_s=-8.,
            assembled_time_s=-8.,objects=(old,)),replace(self.priors,
                carton_size_catalog_m=((.09,.05,.04),tuple(SIZE),(.11,.06,.05))))
        current=replace(old,track_id='new_box',time_s=1.,position_m=tuple(CENTRE+[.004,0,.0025]),
            size_m=(.09,.05,.04),mask_refs=tuple(scene.masks),velocity_m_s=None,
            quality=dict(old.quality,observed_top_z_m=CENTRE[2]+SIZE[2]/2))
        scene=replace(scene,objects=(current,))
        self.assertEqual(reference.confirm(packet,scene)['reason'],'PRECLOSE_REFERENCE_EXPIRED')
        measured=reference.reacquire(packet,scene)
        self.assertIsNotNone(measured)
        self.assertEqual(measured.size_m,old.size_m)
        self.assertEqual(measured.time_s,1.)
        np.testing.assert_allclose(measured.position_m,CENTRE,atol=.001)
        self.assertIsNone(measured.velocity_m_s)
        with_neighbour=replace(scene,objects=(current,replace(current,track_id='another')))
        self.assertIsNone(reference.reacquire(packet,with_neighbour))
        smaller_top=replace(current,quality=dict(current.quality,
            observed_top_z_m=current.quality['observed_top_z_m']-.005))
        self.assertIsNone(reference.reacquire(packet,replace(scene,objects=(smaller_top,))))
        self.assertIsNone(reference.reacquire(self.missing_packet(),scene))

    def test_fragment_merge_retains_other_objects_and_old_anchor_time(self):
        objects=tuple(replace(self.reference,time_s=1.,track_id=self.reference.track_id if i==0 else 'alias',
            mask_refs=(ref,),cameras=(ref.split('/')[0],),posture='UNKNOWN',
            failures=('PARTIAL_OR_MERGED_GEOMETRY',),visibility='PARTIAL_OR_UNCERTAIN')
            for i,ref in enumerate(self.scene.masks))
        other=replace(objects[0],track_id='neighbour',position_m=(.3,.1,.95),mask_refs=('unrelated',))
        scene=replace(self.scene,objects=objects+(other,));est=CartonEstimator(self.priors)
        est._anchors[self.reference.track_id]=self.reference
        proof=self.check.confirm(self.packet,scene)
        result=reconcile_preclose_target(est,scene,proof)
        self.assertEqual(len(result.objects),2);self.assertIn(other,result.objects)
        obj=next(o for o in result.objects if o.track_id==self.reference.track_id)
        self.assertEqual(obj.visibility,'REFERENCE_CONSTRAINED_CUBOID')
        self.assertFalse(pose_is_observed(obj,result));self.assertEqual(obj.posture,'UNKNOWN')
        self.assertIs(result.masks,scene.masks)
        self.assertEqual(est._anchors[self.reference.track_id].time_s,.9)

    def conflict_fragments(self,tape=True):
        packet,scene=synthetic_observation(np.eye(3));cameras={}
        for name,frame in packet.cameras.items():
            if frame is None:cameras[name]=None;continue
            xyz=deproject(frame);rgb=frame.rgb.copy()
            if tape:
                stripe=(np.abs(xyz[...,2]-CENTRE[2]-SIZE[2]/2)<.001)&(np.abs(xyz[...,1]-CENTRE[1])<.006)
                rgb[stripe]=[205,160,100]
            cameras[name]=replace(frame,rgb=rgb)
        objects=[]
        for i,ref in enumerate(scene.masks):
            objects.append(replace(self.reference,time_s=1.,track_id='part_'+str(i),
                position_m=tuple(CENTRE+[.025*i,0,0]),size_m=(.09,.05,.04),posture='UNKNOWN',
                mask_refs=(ref,),cameras=(ref.split('/')[0],),visibility='PARTIAL_OR_UNCERTAIN',
                failures=('CAMERA_SIZE_CONFLICT',) if i==0 else ('PARTIAL_OR_MERGED_GEOMETRY',),
                quality=dict(self.reference.quality,observed_top_z_m=CENTRE[2]+SIZE[2]/2)))
        return replace(packet,cameras=cameras),replace(scene,objects=tuple(objects))

    def test_current_geometry_resolves_size_conflict_and_merges_only_supported_fragments(self):
        p,s=self.conflict_fragments();current=self.check.reacquire(p,s)
        self.assertIsNotNone(current)
        self.assertEqual(current.size_m,tuple(SIZE));self.assertEqual(current.posture,'UPRIGHT')
        self.assertEqual(current.quality['preclose_resolved_size_conflict'],1.)
        self.assertEqual(current.quality['preclose_reacquired_source_tracks'],len(s.objects))
        other=replace(s.objects[0],track_id='other',position_m=(.4,0,.8),mask_refs=('unrelated',))
        s=replace(s,objects=s.objects+(other,));est=CartonEstimator(self.priors)
        updated=reconcile_reacquired_target(est,s,current)
        self.assertEqual(len(updated.objects),2);self.assertIn(other,updated.objects)
        fresh=PreCloseTarget(current,updated,self.priors)
        self.assertTrue(fresh.confirm(p,updated)['supported'])
        self.assertEqual(self.check.obj.time_s,.9)

    def test_conflict_reacquisition_requires_current_packaging_and_fresh_complete_geometry(self):
        p,s=self.conflict_fragments(tape=False)
        self.assertIsNone(self.check.reacquire(p,s))
        p,s=self.conflict_fragments()
        self.assertIsNone(self.check.reacquire(replace(p,assembled_time_s=1.2),s))
        self.assertIsNone(self.check.reacquire(self.missing_packet(),s))
        bad=replace(s,objects=(replace(s.objects[0],quality=dict(s.objects[0].quality,packaging_conflict=1.)),)+s.objects[1:])
        self.assertIsNone(self.check.reacquire(p,bad))
        partial_p,partial_s=self.partial()
        partial_s=replace(partial_s,objects=s.objects)
        self.assertIsNone(self.check.reacquire(partial_p,partial_s))

    def test_reacquired_merge_rejects_stale_proof_and_preserves_unmatched_neighbour(self):
        p,s=self.conflict_fragments();current=self.check.reacquire(p,s)
        est=CartonEstimator(self.priors)
        with self.assertRaisesRegex(RuntimeError,'PROOF_INVALID'):
            reconcile_reacquired_target(est,replace(s,assembled_time_s=1.2),current)
        neighbour=replace(s.objects[0],track_id='independent',position_m=tuple(CENTRE+[.06,0,0]),mask_refs=('unrelated',))
        updated=reconcile_reacquired_target(est,replace(s,objects=s.objects+(neighbour,)),current)
        self.assertIn(neighbour,updated.objects)
        # A separate nearby carton remains an obstacle, not an identity veto.
        PreCloseTarget(current,updated,self.priors)
        duplicate=replace(neighbour,mask_refs=current.mask_refs)
        with self.assertRaises(ValueError):
            PreCloseTarget(current,replace(updated,objects=updated.objects+(duplicate,)),self.priors)

    def test_stage_full_measurement_refreshes_expired_pose_without_retyping_size(self):
        p,s=self.conflict_fragments()
        self.check.obj=replace(self.check.obj,time_s=-10.)
        fresh,proof=self.check.evaluate(p,s)
        self.assertTrue(preclose_supported(proof,1.),proof)
        self.assertIsNot(fresh,self.check)
        self.assertEqual(fresh.obj.size_m,self.reference.size_m)
        self.assertEqual(fresh.obj.time_s,1.)
        self.assertEqual(proof['next_action'],'CONTINUE')
        self.assertEqual(proof['reference_age_s'],0.)
        self.assertEqual(self.check.obj.time_s,-10.)

    def test_stage_partial_measurement_cannot_refresh_expired_hidden_axis(self):
        fresh,proof=self.check.evaluate(self.packet,self.scene)
        self.assertIs(fresh,self.check)
        self.assertEqual(proof['reference_time_s'],.9)
        self.check.obj=replace(self.check.obj,time_s=-10.)
        fresh,proof=self.check.evaluate(self.packet,self.scene)
        self.assertIs(fresh,self.check)
        self.assertFalse(proof['supported'])
        self.assertEqual(proof['next_action'],'OBSERVE')
        self.assertEqual(proof['reason'],'PRECLOSE_REFERENCE_EXPIRED')

    def test_stage_visible_motion_requires_replan(self):
        p,s=self.partial(centre=CENTRE+[0,.006,0])
        _,proof=self.check.evaluate(p,s)
        self.assertFalse(proof['supported'])
        self.assertEqual(proof['next_action'],'REPLAN')

    def test_task_source_query_uses_confirmed_size_and_current_full_fit(self):
        est=CartonEstimator(self.priors,persistent_identities=True)
        original=est._retain_identities(self.reference_scene).objects[0]
        p,s=self.conflict_fragments()
        result=est.resolve_action_sources(p,s,[original.identity_id])
        target=next(o for o in result.objects if o.identity_id==original.identity_id)
        self.assertEqual(target.size_m,original.size_m)
        self.assertEqual(target.time_s,p.observation_time_s)
        self.assertFalse(target.failures)
        self.assertTrue(pose_is_observed(target,result))

    def test_task_source_query_never_publishes_old_hidden_pose(self):
        est=CartonEstimator(self.priors,persistent_identities=True)
        original=est._retain_identities(self.reference_scene).objects[0]
        p,s=self.partial()
        s=replace(s,objects=())
        result=est.resolve_action_sources(p,s,[original.identity_id])
        self.assertFalse(result.objects)

    def test_complete_closed_stage_passes_matching_reference_to_holding(self):
        p,s=self.conflict_fragments()
        fresh,proof=self.check.evaluate(p,s,closed=True)
        self.assertTrue(proof['full_pose_measured'])
        self.assertEqual(fresh.obj.time_s,proof['reference_time_s'])
        tcp=CENTRE+[0,0,.009];r=np.diag([1.,-1.,-1.])
        tracker=CuboidHoldingTracker.from_preclose_support(fresh.obj,proof,tcp,r,self.priors,packet=p)
        self.assertTrue(tracker.allow_grasp_translation)
        with self.assertRaises(ValueError):
            CuboidHoldingTracker.from_preclose_support(self.reference,proof,tcp,r,self.priors,packet=p)

    def test_partial_closed_stage_keeps_matching_original_reference(self):
        fresh,proof=self.check.evaluate(self.packet,self.scene,closed=True)
        self.assertFalse(proof['full_pose_measured'])
        self.assertIs(fresh,self.check)
        self.assertEqual(proof['reference_time_s'],self.reference.time_s)
        tracker=CuboidHoldingTracker.from_preclose_support(fresh.obj,proof,CENTRE+[0,0,.009],
            np.diag([1.,-1.,-1.]),self.priors,packet=self.packet)
        self.assertTrue(tracker.allow_grasp_translation)

    def test_close_initialization_still_needs_current_holding_support(self):
        tcp=CENTRE+[0,0,.009];r=np.diag([1.,-1.,-1.])
        opened=self.check.confirm(self.packet,self.scene)
        with self.assertRaises(ValueError):
            CuboidHoldingTracker.from_preclose_support(self.reference,opened,tcp,r,self.priors)
        closed=self.check.confirm(self.packet,self.scene,closed=True)
        tracker=CuboidHoldingTracker.from_preclose_support(self.reference,closed,tcp,r,self.priors)
        measured=tracker.observe(self.packet,self.scene,tcp,r)
        self.assertTrue(grasp_constrained_cuboid_supported(measured,1.))
        release=HeldSurfaceTracker.from_preclose_support(closed,tcp,r,1.)
        np.testing.assert_allclose(release.offset,tracker.offset)
        self.assertIsNone(tracker.observe(self.missing_packet(),self.scene,tcp,r))

    def missing_packet(self):
        return replace(self.packet,cameras={name:None for name in self.packet.cameras},
            camera_status={name:'MISSING' for name in self.packet.cameras})

    def test_postclose_full_pose_is_reused_only_for_the_same_exposure_once(self):
        p,s=synthetic_observation(np.eye(3))
        check=PreCloseTarget(self.reference,self.reference_scene,self.priors)
        proof=check.confirm(p,s,closed=True)
        self.assertTrue(proof['full_pose_measured'])
        tcp=CENTRE+[0.,0.,.009];rotation=np.diag([1.,-1.,-1.])
        def tracker():
            return CuboidHoldingTracker.from_preclose_support(self.reference,proof,tcp,rotation,self.priors,packet=p)
        t=tracker()
        self.assertTrue(t.observe(p,s,tcp,rotation)['current_postclose_surface_reused'])
        self.assertFalse(t.observe(p,s,tcp,rotation).get('current_postclose_surface_reused',False))
        self.assertIsNone(tracker().observe(self.missing_packet(),s,tcp,rotation))

    def test_current_packaging_conflict_and_ambiguous_reference_rejected(self):
        conflict=replace(self.reference,time_s=1.,posture='SIDE')
        self.assertFalse(self.check.confirm(self.packet,replace(self.scene,objects=(conflict,)))['supported'])
        with self.assertRaises(ValueError):
            PreCloseTarget(self.reference,replace(self.reference_scene,objects=(self.reference,
                replace(self.reference,track_id='nearby'))),self.priors)


if __name__=='__main__':unittest.main()
