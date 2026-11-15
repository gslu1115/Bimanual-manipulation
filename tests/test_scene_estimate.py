"""Synthetic calibrated RGB-D tests. No simulator labels enter the estimator."""
import unittest
from dataclasses import replace
import math
import numpy as np
from workstation.observations.observation_packet import (
    CameraFrame, ObservationPacket, RobotState, FORMAL_CAMERA_NAMES, ROBOT_NAMES)
from workstation.perception.scene_estimate import ScenePriors
from workstation.perception.rgbd_cartons import CartonEstimator, axis_error


def packet(time=1., yaw=.3, centre=(0.,-.10), tape=True, size=(.1,.055,.045),
           empty=False, assembled=None, valid=True):
    h,w = 240,320
    k = np.array([[500.,0.,w/2],[0.,500.,h/2],[0.,0.,1.]])
    t = np.diag([1.,-1.,-1.,1.]); t[:3,3] = [0.,-.1,1.5]
    v,u = np.indices((h,w)); z = 1.5-(.76+size[2])
    xy = np.stack(((u-w/2)*z/500.,-(v-h/2)*z/500.),axis=-1)-centre+[0.,-.1]
    local = xy@np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    mask = (np.abs(local[...,0]) < size[0]/2) & (np.abs(local[...,1]) < size[1]/2)
    if empty: mask[:] = False
    rgb = np.full((h,w,3),100,np.uint8); rgb[mask] = [160,110,65]
    if tape: rgb[mask & (np.abs(local[...,1]) < .006)] = [205,160,100]
    depth = np.full((h,w),1.5-.76,np.float32); depth[mask] = z
    states = {name:RobotState(time,np.zeros(7),np.full(2,.04),np.zeros(7),np.zeros(2)) for name in ROBOT_NAMES}
    frame = CameraFrame('scene_camera',rgb,depth,np.full((h,w),valid,bool),k,t,time,int(time*100),states)
    cameras = {name:frame if name == 'scene_camera' else None for name in FORMAL_CAMERA_NAMES}
    status = {name:'OK' if name == 'scene_camera' else 'MISSING' for name in FORMAL_CAMERA_NAMES}
    return ObservationPacket(cameras,status,states,time if assembled is None else assembled,states,time)


class EstimateTests(unittest.TestCase):
    def test_identity_outlives_pose_but_missing_object_has_no_current_pose(self):
        est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)),persistent_identities=True)
        first=est.estimate(packet())
        key=first.objects[0].identity_id
        missing=est.estimate(packet(time=10.,empty=True))
        self.assertFalse(missing.objects)
        self.assertEqual(missing.identities[0].identity_id,key)
        self.assertEqual(missing.identities[0].last_observed_s,1.)
        current=est.estimate(packet(time=11.))
        self.assertEqual(current.objects[0].identity_id,key)
        self.assertEqual(current.objects[0].time_s,11.)
        self.assertIsNone(current.objects[0].velocity_m_s)

    def test_confirmed_size_is_not_replaced_by_clipped_catalogue_class(self):
        est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)),persistent_identities=True)
        scene=est.estimate(packet())
        original=scene.objects[0]
        changed=replace(original,size_m=(.11,.06,.05),time_s=1.1)
        result=est._retain_identities(replace(scene,objects=(changed,),observation_time_s=1.1))
        self.assertEqual(result.objects[0].size_m,original.size_m)
        self.assertEqual(result.identities[0].size_m,original.size_m)
        self.assertIn('CONFIRMED_SIZE_CONFLICT',result.objects[0].failures)

    def test_delivered_identity_is_not_reassigned_after_occlusion(self):
        est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)),persistent_identities=True)
        key=est.estimate(packet()).objects[0].identity_id
        est.mark_identity(key,'delivered')
        current=est.estimate(packet(time=10.))
        self.assertNotEqual(current.objects[0].identity_id,key)
        self.assertEqual(est._identities[key].state,'delivered')

    def test_next_pick_uses_current_identity_without_requiring_all_sources(self):
        from workstation.planning.pair_task import match_task_sources
        est=CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)),persistent_identities=True)
        first=est.estimate(packet());obj=first.objects[0];key=obj.identity_id
        refs={key:dict(size_m=obj.size_m,position_m=obj.position_m),'hidden':dict(size_m=obj.size_m)}
        current=est.estimate(packet(time=1.1,centre=(.01,-.1)))
        self.assertEqual(set(match_task_sources(current,refs)),{key})
        self.assertFalse(match_task_sources(replace(current,assembled_time_s=2.),refs))

    def test_conflicting_sizes_without_top_rectangle_remain_unknown(self):
        from unittest.mock import patch
        from workstation.perception.rgbd_cartons import deproject
        p=packet(yaw=.5,size=(.12,.065,.055));full=p.cameras['scene_camera']
        xy=deproject(full)[...,:2]-[0.,-.1]
        valid=full.valid_depth & (xy@np.array([math.cos(.5),math.sin(.5)])<.051)
        cameras=dict(p.cameras,scene_camera=replace(full,valid_depth=valid),
                     right_wrist_camera=replace(full,name='right_wrist_camera'))
        p=replace(p,cameras=cameras,camera_status={**p.camera_status,'right_wrist_camera':'OK'})
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16),
            carton_size_catalog_m=((.1,.055,.045),(.11,.06,.05),(.12,.065,.055)))
        with patch('workstation.perception.cuboid_pose.fit_cuboid_pose',
                   return_value=dict(top_rectangle=None,failures=['NO_TOP_RECTANGLE'])) as fit:
            scene=CartonEstimator(priors).estimate(p)
        self.assertTrue(fit.called)
        self.assertEqual(len(scene.objects),1)
        self.assertEqual(scene.objects[0].posture,'UNKNOWN')
        self.assertIn('CAMERA_SIZE_CONFLICT',scene.objects[0].failures)

    def test_clipped_complete_view_cannot_shrink_full_other_view(self):
        from workstation.perception.rgbd_cartons import deproject
        for yaw in (0.,.5):
            p=packet(yaw=yaw,size=(.12,.065,.055))
            full=p.cameras['scene_camera']
            xy=deproject(full)[...,:2]-[0.,-.1]
            axis=np.array([math.cos(yaw),math.sin(yaw)])
            valid=full.valid_depth & (xy@axis<.051)
            cameras=dict(p.cameras)
            cameras['scene_camera']=replace(full,valid_depth=valid)
            cameras['right_wrist_camera']=replace(full,name='right_wrist_camera')
            p=replace(p,cameras=cameras,camera_status={**p.camera_status,'right_wrist_camera':'OK'})
            priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16),
                carton_size_catalog_m=((.1,.055,.045),(.11,.06,.05),(.12,.065,.055)))
            result=CartonEstimator(priors).estimate(p)
            self.assertEqual(len(result.objects),1)
            obj=result.objects[0]
            if yaw==0.:
                # This coarse raster lacks the fitter's full extent proof.
                self.assertIn('CAMERA_SIZE_CONFLICT',obj.failures)
                self.assertEqual(obj.posture,'UNKNOWN')
                continue
            self.assertEqual(obj.size_m,(.12,.065,.055))
            self.assertFalse(obj.failures)
            self.assertLess(np.linalg.norm(np.asarray(obj.position_m)-[0.,-.1,.7875]),.003)

    def setUp(self):
        self.estimator = CartonEstimator(ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16)))

    def test_metric_pose_and_tape(self):
        scene = self.estimator.estimate(packet())
        self.assertEqual(len(scene.objects),1)
        obj = scene.objects[0]
        self.assertLess(np.linalg.norm(np.array(obj.position_m)-[0.,-.1,.7825]),.003)
        self.assertLess(axis_error(obj.axis_yaw_rad,.3),math.radians(2))
        self.assertEqual(obj.posture,'UPRIGHT')
        self.assertFalse(obj.failures)
        self.assertTrue(obj.track_id.startswith('visual_'))
        self.assertEqual(scene.frame,'workcell')
        self.assertNotIn('masks',scene.to_dict())

    def test_hidden_top_not_inverted(self):
        obj = self.estimator.estimate(packet(tape=False)).objects[0]
        self.assertEqual(obj.posture,'UNKNOWN')
        self.assertEqual({p.label for p in obj.hypotheses},{'UPRIGHT','INVERTED'})

    def test_warm_grey_gripper_not_in_carton_mask(self):
        p=packet(yaw=0.)
        f=p.cameras['scene_camera']; rgb=f.rgb.copy(); depth=f.depth_m.copy()
        # Connected warm grey patch, below the carton top, like the real wrist
        # image. It passes the broad brown gate, but is not cardboard/tape.
        patch=np.zeros(depth.shape,bool); patch[99:113,193:207]=True
        rgb[patch]=[114,99,84]; depth[patch]=1.5-.795
        # Actual pale tape has weak chroma too; it must survive on the top plane.
        tape=(rgb[...,0]==205); rgb[tape]=[205,190,155]
        cameras=dict(p.cameras); cameras['scene_camera']=replace(f,rgb=rgb,depth_m=depth)
        scene=self.estimator.estimate(replace(p,cameras=cameras))
        self.assertEqual(len(scene.objects),1)
        mask=scene.masks[scene.objects[0].mask_refs[0]]
        self.assertFalse(mask[patch].any())
        self.assertTrue(mask[tape].all())
        self.assertEqual(scene.objects[0].posture,'UPRIGHT')

    def test_pale_top_under_lighting_still_has_metric_geometry(self):
        p=packet(yaw=0.)
        f=p.cameras['scene_camera']; rgb=f.rgb.copy()
        brown=rgb[...,0]==160; tape=rgb[...,0]==205
        rgb[brown]=[165,145,110]; rgb[tape]=[205,190,155]
        cameras=dict(p.cameras); cameras['scene_camera']=replace(f,rgb=rgb)
        scene=self.estimator.estimate(replace(p,cameras=cameras))
        self.assertEqual(len(scene.objects),1)
        self.assertFalse(scene.objects[0].failures)
        self.assertEqual(scene.objects[0].posture,'UPRIGHT')

    def test_tracking_and_visual_velocity(self):
        first = self.estimator.estimate(packet()).objects[0]
        second = self.estimator.estimate(packet(time=1.2,centre=(.006,-.1))).objects[0]
        self.assertEqual(first.track_id,second.track_id)
        self.assertAlmostEqual(second.velocity_m_s[0],.03,delta=.01)

    def test_elevated_clipped_face_does_not_prove_a_tall_side_posture(self):
        p=packet(tape=False,size=(.055,.045,.1));frame=p.cameras['scene_camera']
        t=frame.T_workcell_from_camera_cv.copy();t[2,3]+=.18
        cameras=dict(p.cameras);cameras['scene_camera']=replace(frame,T_workcell_from_camera_cv=t)
        obj=self.estimator.estimate(replace(p,cameras=cameras)).objects[0]
        self.assertEqual(obj.posture,'UNKNOWN');self.assertEqual(obj.visibility,'PARTIAL_OR_UNCERTAIN')
        self.assertIn('VERTICAL_EXTENT_UNRESOLVED',obj.failures)
        self.assertGreaterEqual(obj.uncertainty_m,.04)
        self.assertIn('SIDE_UNRESOLVED',{h.label for h in obj.hypotheses})

    def test_uncertain_centres_do_not_generate_observed_velocity(self):
        first=self.estimator.estimate(packet()).objects[0]
        partial=self.estimator.estimate(packet(time=1.2,tape=False,size=(.055,.045,.045))).objects[0]
        self.assertEqual(first.track_id,partial.track_id);self.assertTrue(partial.failures)
        self.assertIsNone(partial.velocity_m_s)
        current=self.estimator.estimate(packet(time=1.4)).objects[0]
        self.assertIsNone(current.velocity_m_s)
        fresh=self.estimator.estimate(packet(time=1.6)).objects[0]
        self.assertIsNotNone(fresh.velocity_m_s)

    def test_stale_returns_no_target(self):
        scene = self.estimator.estimate(packet(assembled=1.3))
        self.assertFalse(scene.objects)
        self.assertIn('scene_camera:STALE',scene.failures)

    def test_invalid_depth_returns_no_target(self):
        self.assertFalse(self.estimator.estimate(packet(valid=False)).objects)

    def test_empty_returns_no_target(self):
        self.assertIn('NO_CARTON_OBSERVED',self.estimator.estimate(packet(empty=True)).failures)

    def test_side_is_explicitly_unresolved(self):
        obj = self.estimator.estimate(packet(size=(.1,.045,.055),tape=False)).objects[0]
        self.assertEqual(obj.posture,'SIDE')
        self.assertEqual(obj.hypotheses[0].label,'SIDE_UNRESOLVED')

    def test_merged_rejected(self):
        obj = self.estimator.estimate(packet(size=(.20,.055,.045))).objects[0]
        self.assertIn('PARTIAL_OR_MERGED_GEOMETRY',obj.failures)

    def test_hand_occlusion_fragments_share_visual_track(self):
        self.estimator.estimate(packet(yaw=0.))
        p=packet(time=1.2,yaw=0.)
        f=p.cameras['scene_camera']; rgb=f.rgb.copy()
        rgb[:,153:167]=[150,150,150]  # Robot occludes middle, both ends remain visible.
        f=replace(f,rgb=rgb)
        cameras=dict(p.cameras); cameras['scene_camera']=f
        scene=self.estimator.estimate(replace(p,cameras=cameras))
        self.assertEqual(len(scene.objects),1)
        obj=scene.objects[0]
        self.assertEqual(obj.track_id,'visual_0001')
        self.assertFalse(obj.failures)
        self.assertGreaterEqual(obj.quality['visible_fragments_joined'],2)
        self.assertLess(np.linalg.norm(np.array(obj.position_m)-[0.,-.1,.7825]),.003)

    def test_incomplete_fragment_does_not_become_side(self):
        self.estimator.estimate(packet(yaw=0.))
        p=packet(time=1.2,yaw=0.)
        f=p.cameras['scene_camera']; rgb=f.rgb.copy(); rgb[:,165:]=[150,150,150]
        cameras=dict(p.cameras); cameras['scene_camera']=replace(f,rgb=rgb)
        scene=self.estimator.estimate(replace(p,cameras=cameras))
        self.assertTrue(scene.objects[0].failures)
        self.assertEqual(scene.objects[0].posture,'UNKNOWN')

    def test_cross_view_lighting_cannot_manufacture_tape_contrast(self):
        self.estimator.estimate(packet(yaw=0.))
        p=packet(time=1.2,yaw=0.,tape=False)
        f=p.cameras['scene_camera'];rgb=np.full(f.rgb.shape,150,np.uint8)
        # Another view sees only a uniformly bright central patch, without
        # flanking cardboard. Pooling it with the darker first view's flanks
        # would manufacture apparent tape contrast absent from either view.
        centre=(f.depth_m<.72)&(np.abs(np.indices(f.depth_m.shape)[1]-160)<4)
        rgb[centre]=[205,160,100]
        wrist=replace(f,name='left_wrist_camera',rgb=rgb)
        cameras=dict(p.cameras);cameras['left_wrist_camera']=wrist
        status=dict(p.camera_status);status['left_wrist_camera']='OK'
        scene=self.estimator.estimate(replace(p,cameras=cameras,camera_status=status))
        self.assertEqual(len(scene.objects),1)
        self.assertEqual(scene.objects[0].posture,'UNKNOWN')
        self.assertEqual(scene.objects[0].quality['tape_camera_count'],0.)

    def test_conflicting_camera_posture_cues_are_not_forced_upright(self):
        obj=self.estimator.estimate(packet(yaw=0.)).objects[0]
        p=packet(yaw=0.);f=p.cameras['scene_camera']
        cameras=dict(p.cameras);cameras['left_wrist_camera']=replace(f,name='left_wrist_camera')
        status=dict(p.camera_status);status['left_wrist_camera']='OK'
        estimator=CartonEstimator(self.estimator.priors)
        def detections(frame,*args):
            ref=f'{frame.name}/{frame.sequence_id}/component_1'
            return [replace(obj,posture='UPRIGHT' if frame.name=='scene_camera' else 'INVERTED',
                            mask_refs=(ref,),cameras=(frame.name,))],{ref:frame.depth_m<.72}
        estimator._detect=detections
        scene=estimator.estimate(replace(p,cameras=cameras,camera_status=status))
        self.assertEqual(scene.objects[0].posture,'UNKNOWN')
        self.assertIn('PACKAGING_CUE_CONFLICT',scene.objects[0].failures)
        self.assertEqual({h.label for h in scene.objects[0].hypotheses},{'UPRIGHT','INVERTED'})

    def test_mismatched_state_rejected(self):
        p = packet()
        states = {k:replace(v,sample_time_s=1.04) for k,v in p.robot_state.items()}
        scene = self.estimator.estimate(replace(p,robot_state=states))
        self.assertFalse(scene.objects)
        self.assertIn('scene_camera:UNSYNCED',scene.failures)

    def test_latest_does_not_replace_matched(self):
        p = packet()
        latest = {k:replace(v,sample_time_s=1.1,joint_positions_rad=np.ones(7)) for k,v in p.robot_state.items()}
        self.assertEqual(len(self.estimator.estimate(replace(p,latest_robot_state=latest)).objects),1)

    def test_backwards_time_rejected(self):
        self.estimator.estimate(packet())
        with self.assertRaises(ValueError): self.estimator.estimate(packet(time=.8))

    def test_prior_allowlist(self):
        from pathlib import Path
        import json
        config = json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text())
        priors = ScenePriors.from_config(config)
        self.assertFalse(hasattr(priors,'box_xy'))
        self.assertFalse(hasattr(priors,'seed'))


if __name__ == '__main__': unittest.main()
