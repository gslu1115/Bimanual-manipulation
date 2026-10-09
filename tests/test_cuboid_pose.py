"""Calibrated synthetic RGB-D cuboids, including rotated and ambiguous boxes."""
import math
import unittest
from dataclasses import replace
import numpy as np
from workstation.observations.observation_packet import (
    CameraFrame,ObservationPacket,RobotState,FORMAL_CAMERA_NAMES,ROBOT_NAMES)
from workstation.perception.scene_estimate import ScenePriors,SceneEstimate
from workstation.perception.cuboid_pose import fit_cuboid_pose,measured_cuboid_supported
from workstation.perception.held_surface import CuboidHoldingTracker
from workstation.perception.scene_estimate import ObjectEstimate,PoseHypothesis
from workstation.observations.camera_geometry import quaternion_from_matrix
from workstation.planning.observed_workspace import ObservedWorkspace
from workstation.simulation.robot_driver import RobotMotion
from workstation.perception.rgbd_cartons import CartonEstimator


SIZE=np.array([.1,.055,.045])
CENTRE=np.array([0.,-.1,.95])
PRIORS=ScenePriors(tuple(SIZE),.76,(-.38,.38,-.51,.16))


def turn(axis,angle):
    angle=math.radians(angle);c,s=math.cos(angle),math.sin(angle)
    if axis==0:return np.array([[1.,0,0],[0,c,-s],[0,s,c]])
    if axis==1:return np.array([[c,0,s],[0,1.,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def synthetic_observation(rotation,centre=CENTRE,views=None):
    # Ray/cuboid intersections generate an optical depth image. Tests use no
    # simulator, labels or mocked deprojection. These are not physical trials.
    h,w=240,320;k=np.array([[520.,0,w/2],[0,520.,h/2],[0,0,1.]])
    states={name:RobotState(1.,np.zeros(7),np.full(2,.025)) for name in ROBOT_NAMES}
    views=views or ([0.,-.45,1.28],[.32,-.16,1.12],[-.28,-.08,1.08])
    cameras={name:None for name in FORMAL_CAMERA_NAMES};statuses={name:'MISSING' for name in FORMAL_CAMERA_NAMES};masks={}
    for name,origin in zip(FORMAL_CAMERA_NAMES,views):
        origin=np.asarray(origin,float);forward=np.asarray(centre)-origin;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right)
        t=np.eye(4);t[:3,:3]=np.column_stack((right,down,forward));t[:3,3]=origin
        v,u=np.indices((h,w));rays=np.stack(((u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],np.ones((h,w))),axis=-1)@t[:3,:3].T
        local_origin=(origin-centre)@rotation;local_ray=rays@rotation
        with np.errstate(divide='ignore',invalid='ignore'):
            a=(-SIZE/2-local_origin)/local_ray;b=(SIZE/2-local_origin)/local_ray
        near=np.minimum(a,b).max(axis=-1);far=np.maximum(a,b).min(axis=-1)
        mask=(far>=near)&(near>0)&np.isfinite(near)
        depth=np.where(mask,near,2.).astype(np.float32);rgb=np.full((h,w,3),90,np.uint8);rgb[mask]=[160,110,65]
        frame=CameraFrame(name,rgb,depth,np.ones((h,w),bool),k,t,1.,100,states)
        cameras[name]=frame;statuses[name]='OK';masks[name+'/100/carton']=mask
    packet=ObservationPacket(cameras,statuses,states,1.,states,1.)
    scene=SceneEstimate(1.,1.,'workcell',(),(),statuses,masks)
    return packet,scene


class CuboidPoseTests(unittest.TestCase):
    def test_rotating_geometry_is_measured_in_metric_workcell(self):
        for angle in (0.,35.,90.,135.,180.):
            with self.subTest(angle=angle):
                r=turn(2,23)@turn(0,angle);p,s=synthetic_observation(r)
                # Deliberately offset the hint: the returned pose must follow RGB-D.
                result=fit_cuboid_pose(p,s,CENTRE+[.002,-.001,.001],r@turn(2,2),SIZE,PRIORS)
                self.assertEqual(result['status'],'GEOMETRY_OBSERVED',result)
                np.testing.assert_allclose(result['position_m'],CENTRE,atol=.0015)
                delta=np.asarray(result['rotation']).T@r
                error=math.degrees(math.acos(np.clip((np.trace(delta)-1)/2,-1,1)))
                self.assertLess(error,.5)
                self.assertTrue(measured_cuboid_supported(result,1.))
                self.assertFalse(result['semantic_pose_unique'])

    def test_side_rotation_about_long_and_short_axes(self):
        for r in (turn(0,90),turn(1,90),turn(1,45)@turn(2,32)):
            p,s=synthetic_observation(r)
            result=fit_cuboid_pose(p,s,CENTRE,r,SIZE,PRIORS)
            self.assertEqual(result['status'],'GEOMETRY_OBSERVED',result)
            np.testing.assert_allclose(result['position_m'],CENTRE,atol=.0015)

    def test_opposite_semantic_hypotheses_remain_geometrically_equivalent(self):
        r=turn(0,90);p,s=synthetic_observation(r)
        for hint in (r,r@turn(0,180)):
            result=fit_cuboid_pose(p,s,CENTRE,hint,SIZE,PRIORS)
            self.assertEqual(result['status'],'GEOMETRY_OBSERVED',result)
            self.assertFalse(result['semantic_pose_unique'])

    def test_full_single_face_can_supply_tangent_axes(self):
        p,s=synthetic_observation(np.eye(3),views=([0.,-.1001,1.35],))
        result=fit_cuboid_pose(p,s,CENTRE,np.eye(3),SIZE,PRIORS)
        self.assertEqual(result['status'],'GEOMETRY_OBSERVED',result)
        self.assertEqual(result['observed_face_axes'],[2])
        np.testing.assert_allclose(result['position_m'],CENTRE,atol=.0015)

    def test_partial_single_face_cannot_inherit_hidden_edges_from_hint(self):
        p,s=synthetic_observation(np.eye(3),views=([0.,-.1001,1.35],))
        masks={ref:mask.copy() for ref,mask in s.masks.items()}
        for mask in masks.values():mask[:,:155]=False;mask[:,180:]=False
        result=fit_cuboid_pose(p,replace(s,masks=masks),CENTRE,np.eye(3),SIZE,PRIORS)
        self.assertEqual(result['status'],'UNKNOWN');self.assertNotIn('position_m',result)

    def test_whole_merged_instances_are_not_cropped_to_the_hint(self):
        p,s=synthetic_observation(turn(0,40))
        masks={ref:mask.copy() for ref,mask in s.masks.items()}
        for mask in masks.values():mask[20:50,20:50]=True
        result=fit_cuboid_pose(p,replace(s,masks=masks),CENTRE,turn(0,40),SIZE,PRIORS)
        self.assertEqual(result['status'],'UNKNOWN');self.assertNotIn('position_m',result)

    def test_isolated_mask_outliers_do_not_define_measured_face_extents(self):
        r=turn(0,40);p,s=synthetic_observation(r)
        masks={ref:mask.copy() for ref,mask in s.masks.items()}
        for mask in masks.values():mask[20:28,20:28]=True
        result=fit_cuboid_pose(p,replace(s,masks=masks),CENTRE,r,SIZE,PRIORS)
        self.assertEqual(result['status'],'GEOMETRY_OBSERVED',result)
        self.assertGreaterEqual(result['surface_inlier_ratio'],.95)
        np.testing.assert_allclose(result['position_m'],CENTRE,atol=.0015)

    def test_visible_stationary_box_does_not_follow_a_rising_tcp_hint(self):
        p,s=synthetic_observation(np.eye(3))
        result=fit_cuboid_pose(p,s,CENTRE+[0.,0.,.05],np.eye(3),SIZE,PRIORS)
        self.assertEqual(result['status'],'UNKNOWN');self.assertNotIn('position_m',result)

    def test_stale_scene_and_packet_rejected(self):
        p,s=synthetic_observation(np.eye(3))
        for packet,scene in ((replace(p,assembled_time_s=1.3),s),
                             (p,replace(s,observation_time_s=.9)),(p,replace(s,frame='camera'))):
            result=fit_cuboid_pose(packet,scene,CENTRE,np.eye(3),SIZE,PRIORS)
            self.assertEqual(result['status'],'UNKNOWN')

    def test_unsynchronized_old_frames_cannot_be_relabelled(self):
        p,s=synthetic_observation(np.eye(3));cameras={}
        for name,f in p.cameras.items():
            states={arm:replace(state,sample_time_s=.8) for arm,state in f.robot_state_at_frame.items()}
            cameras[name]=replace(f,sample_time_s=.8,robot_state_at_frame=states)
        result=fit_cuboid_pose(replace(p,cameras=cameras),s,CENTRE,np.eye(3),SIZE,PRIORS)
        self.assertEqual(result['status'],'UNKNOWN')

    def test_invalid_depth_or_missing_status_is_not_geometry(self):
        p,s=synthetic_observation(np.eye(3));cameras={name:replace(f,valid_depth=np.zeros_like(f.valid_depth)) for name,f in p.cameras.items()}
        result=fit_cuboid_pose(replace(p,cameras=cameras),s,CENTRE,np.eye(3),SIZE,PRIORS)
        self.assertEqual(result['status'],'UNKNOWN')
        result=fit_cuboid_pose(p,replace(s,camera_status={name:'INVALID' for name in FORMAL_CAMERA_NAMES}),CENTRE,np.eye(3),SIZE,PRIORS)
        self.assertEqual(result['status'],'UNKNOWN')

    def test_invalid_hint_and_unbounded_depth_margin(self):
        p,s=synthetic_observation(np.eye(3))
        for centre,r,size in (([0,0],np.eye(3),SIZE),(CENTRE,np.diag([1,1,-1]),SIZE),(CENTRE,np.eye(3),[-1,.055,.045])):
            with self.assertRaises(ValueError):fit_cuboid_pose(p,s,centre,r,size,PRIORS)
        result=fit_cuboid_pose(p,s,CENTRE,np.eye(3),SIZE,replace(PRIORS,optical_depth_error_margin_m=.01))
        self.assertEqual(result['status'],'UNKNOWN')

    def test_payload_proof_needs_all_measured_coordinates(self):
        p,s=synthetic_observation(np.eye(3));result=fit_cuboid_pose(p,s,CENTRE,np.eye(3),SIZE,PRIORS)
        self.assertTrue(measured_cuboid_supported(result,1.))
        for changes in ({'time_s':.9},{'uncertainty_m':.009},{'surface_inlier_ratio':.94},
                        {'coordinate_constraints':[]},{'rotation':np.diag([1,1,-1]).tolist()},
                        {'geometric_pose_measured':False},{'status':'UNKNOWN'}):
            self.assertFalse(measured_cuboid_supported(dict(result,**changes),1.))


def visual_object(scene,r,ambiguous=False):
    rotations=(r,r@turn(0,180)) if ambiguous else (r,)
    hypotheses=tuple(PoseHypothesis('SIDE' if ambiguous else 'UPRIGHT',
        tuple(quaternion_from_matrix(value)),'initial visual geometric hypothesis') for value in rotations)
    return ObjectEstimate('visual_1',1.,tuple(CENTRE),0.,tuple(SIZE),'known dimensions',
        'SIDE' if ambiguous else 'UPRIGHT',hypotheses,tuple(scene.masks),tuple(scene.camera_status),
        'COMPLETE_TOP',1.,{'footprint_coverage':.9},(),uncertainty_m=.003)


class CuboidHoldingIntegrationTests(unittest.TestCase):
    def test_current_rotated_geometry_and_semantic_alternatives_are_retained(self):
        p,s=synthetic_observation(turn(0,90));obj=visual_object(s,turn(0,90),True)
        tracker=CuboidHoldingTracker(obj,CENTRE,np.eye(3),SIZE,PRIORS)
        r=turn(2,18)@turn(0,35);p,s=synthetic_observation(r,CENTRE+[0,0,.04])
        tcp_rotation=r@turn(0,-90)
        result=tracker.observe(p,s,CENTRE+[.002,0,.04],tcp_rotation)
        self.assertIsNotNone(result,tracker.last_failure)
        self.assertEqual(result['top_bottom_semantics'],'UNRESOLVED')
        self.assertEqual(len(result['pose_hypotheses']),2)
        self.assertTrue(result['full_6d_pose_measured']);self.assertFalse(result['semantic_pose_unique'])
        self.assertAlmostEqual(result['height_gain_m'],.04,delta=.0015)
        np.testing.assert_allclose(result['position_m'],CENTRE+[0,0,.04],atol=.0015)

    def test_missing_current_geometry_does_not_return_a_robot_prediction(self):
        p,s=synthetic_observation(np.eye(3));tracker=CuboidHoldingTracker(visual_object(s,np.eye(3)),CENTRE,np.eye(3),SIZE,PRIORS)
        self.assertIsNone(tracker.observe(p,s,CENTRE+[0,0,.05],np.eye(3)))
        self.assertEqual(tracker.last_failure['status'],'UNKNOWN')
        self.assertNotIn('position_m',tracker.last_failure)

    def test_visual_prior_uncertainty_is_not_replaced_by_perfect_synthetic_planes(self):
        p,s=synthetic_observation(np.eye(3));obj=replace(visual_object(s,np.eye(3)),uncertainty_m=.007)
        tracker=CuboidHoldingTracker(obj,CENTRE,np.eye(3),SIZE,PRIORS)
        result=tracker.observe(p,s,CENTRE,np.eye(3));self.assertIsNotNone(result)
        self.assertEqual(result['uncertainty_m'],.007)

    def test_invalid_or_partial_visual_latch_rejected(self):
        p,s=synthetic_observation(np.eye(3));obj=visual_object(s,np.eye(3))
        for changes in ({'uncertainty_m':float('nan')},{'uncertainty_m':.009},
                        {'failures':('OCCLUDED',)},{'visibility':'PARTIAL_OR_UNCERTAIN'},{'hypotheses':()}):
            with self.assertRaises(ValueError):CuboidHoldingTracker(replace(obj,**changes),CENTRE,np.eye(3),SIZE,PRIORS)

    def test_current_measured_geometry_reaches_workspace_and_future_sweep_relation(self):
        r=turn(0,35);p,s=synthetic_observation(r);obj=visual_object(s,r)
        tracker=CuboidHoldingTracker(obj,CENTRE,np.eye(3),SIZE,PRIORS)
        result=tracker.observe(p,s,CENTRE+[.002,0,0],np.eye(3));self.assertIsNotNone(result)
        ws=ObservedWorkspace(p,PRIORS,replace(s,objects=(obj,)))
        driver=RobotMotion.__new__(RobotMotion);driver.observed_workspace=ws
        driver.held_visual_geometry=dict(target_id='visual_1',arm='panda_left',offset=tracker.offset.copy())
        driver.read_states=lambda:(_ for _ in ()).throw(AssertionError('latest robot state must not be read'))
        driver.fk=lambda arm,q:(CENTRE+[.002,0,0],np.eye(3))
        self.assertTrue(driver.confirm_payload_observation('visual_1',tracker,result,p))
        np.testing.assert_allclose(ws.payload_observation['centre'],result['position_m'])
        np.testing.assert_allclose(driver.held_visual_geometry['offset'],np.asarray(result['position_m'])-(CENTRE+[.002,0,0]))
        np.testing.assert_array_equal(tracker.offset,np.zeros(3))
        self.assertEqual(driver.held_visual_geometry['measured_pose_time_s'],1.)
        self.assertFalse(driver.confirm_payload_observation('visual_1',tracker,
            dict(result,coordinate_constraints=[]),p))
        self.assertIsNone(ws.payload_observation)

    def test_workspace_rejects_substituted_prediction_and_incomplete_measured_proof(self):
        r=turn(0,35);p,s=synthetic_observation(r);result=fit_cuboid_pose(p,s,CENTRE+[.002,0,0],r,SIZE,PRIORS)
        ws=ObservedWorkspace(p,PRIORS,s);pos=result['position_m'];rot=result['rotation'];u=result['uncertainty_m']
        self.assertTrue(ws.confirm_payload('visual_1',pos,rot,SIZE,result,u))
        for centre,proof,uncertainty in ((CENTRE+[.002,0,0],result,u),
            (pos,dict(result,coordinate_constraints=[]),u),(pos,result,.003)):
            self.assertFalse(ws.confirm_payload('visual_1',centre,rot,SIZE,proof,uncertainty))
            self.assertIsNone(ws.payload_observation)

    def test_measured_pose_can_reconcile_alias_without_fabricating_scene_semantics(self):
        r=turn(0,35);p,s=synthetic_observation(r);obj=visual_object(s,r)
        alias=replace(obj,track_id='visual_alias',posture='UNKNOWN',failures=('PARTIAL_GEOMETRY',))
        scene=replace(s,objects=(alias,));result=fit_cuboid_pose(p,scene,CENTRE,r,SIZE,PRIORS)
        estimator=CartonEstimator(PRIORS);associated=estimator.associate_supported_target(scene,obj.track_id,result)
        self.assertEqual(associated.objects[0].track_id,obj.track_id)
        self.assertEqual(associated.objects[0].posture,'UNKNOWN')
        self.assertEqual(associated.objects[0].failures,alias.failures)
        unsupported=estimator.associate_supported_target(scene,obj.track_id,dict(result,coordinate_constraints=[]))
        self.assertIs(unsupported,scene)


if __name__=='__main__':unittest.main()
