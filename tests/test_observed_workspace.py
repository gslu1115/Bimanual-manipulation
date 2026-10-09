import unittest
from types import SimpleNamespace
import numpy as np
from workstation.perception.scene_estimate import ScenePriors
from workstation.planning.collision import AABB
from workstation.planning.observed_workspace import ObservedWorkspace


class WorkspaceTests(unittest.TestCase):
    def test_calibrated_margin_still_rejects_behind_surface_and_invalid_depth(self):
        from dataclasses import replace
        ws=self.workspace();frame=next(ws._frames())
        point=np.array([[0.,0.,.802]]) # optical z .998, 2mm ahead of surface
        self.assertFalse(ws._ray_free(point,frame)[0])
        ws.priors=replace(ws.priors,optical_depth_error_margin_m=.001)
        self.assertTrue(ws._ray_free(point,frame)[0])
        self.assertFalse(ws._ray_free(np.array([[0.,0.,.798]]),frame)[0])
        invalid=self.workspace(valid=False);invalid.priors=ws.priors
        self.assertFalse(invalid._ray_free(point,next(invalid._frames()))[0])

    def test_zero_depth_error_margin_is_not_allowed(self):
        with self.assertRaises(ValueError):
            ScenePriors((.04,.02,.02),.8,(-.05,.05,-.05,.05),optical_depth_error_margin_m=0.)

    def workspace(self,depth=1.,valid=True,status='OK'):
        t=np.diag([1.,-1.,-1.,1.]);t[:3,3]=[0.,0.,1.8]
        frame=SimpleNamespace(sample_time_s=1.,robot_state_at_frame={},
            depth_m=np.full((50,50),depth,np.float32),valid_depth=np.full((50,50),valid,bool),
            T_workcell_from_camera_cv=t,K=np.array([[50.,0,25],[0,50.,25],[0,0,1.]]))
        packet=SimpleNamespace(cameras={'camera':frame},camera_status={'camera':status},
            observation_time_s=1.,assembled_time_s=1.)
        priors=ScenePriors((.04,.02,.02),.8,(-.05,.05,-.05,.05))
        table=AABB('table',(0,0,.75),(1.,1.,.1))
        return ObservedWorkspace(packet,priors,SimpleNamespace(objects=()),(table,),resolution=.02)

    def test_clear_depth_and_support_evidence_allows_slot(self):
        ws=self.workspace();self.assertTrue(ws.slot((0,0))['free'])

    def test_no_detection_is_not_empty_when_depth_missing(self):
        ws=self.workspace(valid=False)
        self.assertEqual(ws.slot((0,0))['reason'],'UNOBSERVED_PATH_SPACE')

    def test_unsegmented_foreground_blocks_slot_and_path(self):
        ws=self.workspace(.94)
        self.assertEqual(ws.check_sphere((0,0,.86),.02),'OBSERVED_PATH_OBSTACLE')
        self.assertFalse(ws.slot((0,0))['free'])

    def test_observed_obstacle_gate_does_not_relabel_unknown_as_free(self):
        ws=self.workspace(valid=False)
        self.assertIsNone(ws.check_observed_sphere_obstacles((0,0,.95),.02))
        self.assertTrue(ws.unknown.all())
        occupied=self.workspace(.94)
        self.assertEqual(occupied.check_observed_sphere_obstacles((0,0,.86),.02),'OBSERVED_PATH_OBSTACLE')

    def test_future_box_obstacle_check_keeps_unknown_map_and_measured_obstacles(self):
        ws=self.workspace(valid=False)
        self.assertIsNone(ws.check_observed_box_obstacles((0,0,.95),(.04,.04,.04)))
        self.assertTrue(ws.unknown.all())
        occupied=self.workspace(.94)
        self.assertEqual(occupied.check_observed_box_obstacles((0,0,.86),(.04,.04,.04)),'OBSERVED_PATH_OBSTACLE')

    def test_space_behind_observed_surface_is_unknown(self):
        ws=self.workspace(.8)
        self.assertEqual(ws.check_sphere((0,0,.90),.02),'UNOBSERVED_PATH_SPACE')

    def test_stale_camera_never_clears_space(self):
        ws=self.workspace();ws.packet.assembled_time_s=1.3
        self.assertEqual(ws.check_box((0,0,.95),(.03,.03,.03)),'UNOBSERVED_PATH_SPACE')

    def test_invalid_camera_never_clears_space(self):
        self.assertFalse(self.workspace(status='INVALID').slot((0,0))['free'])

    def test_only_current_robot_model_accounts_for_its_occlusion(self):
        ws=self.workspace(valid=False);ws.robots=((8,np.array([0,0,.95]),.04),)
        self.assertIsNone(ws.check_sphere((0,0,.95),.005))
        self.assertEqual(ws.check_sphere((0,0,1.05),.005),'UNOBSERVED_PATH_SPACE')

    def test_current_robot_boundary_does_not_clear_new_swept_space(self):
        ws=self.workspace(valid=False);ws.robots=((7,np.array([0,0,.95]),.04),)
        self.assertIsNone(ws.check_sphere((0,0,.95),.04))
        self.assertEqual(ws.check_sphere((.01,0,.95),.04),'UNOBSERVED_PATH_SPACE')

    def test_robot_lifting_out_of_workspace_does_not_enter_unknown_space(self):
        ws=self.workspace(valid=False);ws.robots=((7,np.array([0,0,1.45]),.08),)
        self.assertIsNone(ws.check_sphere((0,0,1.48),.08))
        self.assertEqual(ws.check_sphere((.03,0,1.45),.08),'UNOBSERVED_PATH_SPACE')

    def test_unknown_outside_body_in_mixed_cell_does_not_block_body(self):
        ws=self.workspace();ws._ray_free=lambda p,frame:p[:,0]<.001
        self.assertIsNone(ws.check_sphere((-.02,0,.95),.02))
        self.assertEqual(ws.check_sphere((-.017,0,.95),.02),'UNOBSERVED_PATH_SPACE')

    def test_unknown_report_explains_occlusion_in_camera_coordinates(self):
        ws=self.workspace(.8);ws.check_sphere((0,0,.90),.02)
        report=ws.report();self.assertIsNotNone(report['last_unknown_sample_m'])
        self.assertEqual(report['unknown_point_views'][0]['reason'],'BEHIND_OR_TOO_CLOSE_TO_SURFACE')

    def cube(self):
        from workstation.planning.robot_geometry import ConvexLink
        return ConvexLink(np.c_[np.r_[np.eye(3),-np.eye(3)],np.full(6,-.01)],
                          np.full(3,-.01),np.full(3,.01),'test')

    def test_official_body_never_clears_adjacent_sphere_proxy_space(self):
        ws=self.workspace(valid=False);p=np.array([0,0,.95]);model=self.cube()
        ws.robots=((5,p,.06),);ws.robot_links=((model,p,np.eye(3)),)
        self.assertFalse(ws._modelled(np.array([[.04,0,.95]]))[0])
        self.assertTrue(ws._modelled(np.array([[0,0,.95]]))[0])

    def test_held_oriented_box_does_not_query_empty_enclosing_corners(self):
        ws=self.workspace();angle=np.pi/4;c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        ws._ray_free=lambda p,frame:np.abs(p[:,0])+np.abs(p[:,1])<.023
        self.assertIsNone(ws.check_oriented_box((0,0,.95),(.02,.02,.02),rotation))
        self.assertEqual(ws.check_box((0,0,.95),(.02*np.sqrt(2),.02*np.sqrt(2),.02)),
                         'UNOBSERVED_PATH_SPACE')
        self.assertEqual(ws.check_oriented_box((.02,0,.95),(.02,.02,.02),rotation),'UNOBSERVED_PATH_SPACE')

    def test_held_oriented_box_still_requires_interior_evidence_and_blocks_obstacles(self):
        ws=self.workspace(valid=False)
        self.assertEqual(ws.check_oriented_box((0,0,.95),(.02,.02,.02),np.eye(3)),'UNOBSERVED_PATH_SPACE')
        self.assertIsNone(ws.check_oriented_box((0,0,.95),(.02,.02,.02),np.eye(3),check_unknown=False))
        self.assertTrue(ws.unknown.all())
        occupied=self.workspace(.94)
        self.assertEqual(occupied.check_oriented_box((0,0,.86),(.04,.04,.04),np.eye(3),check_unknown=False),
                         'OBSERVED_PATH_OBSTACLE')

    def test_held_oriented_box_rejects_invalid_geometry(self):
        ws=self.workspace()
        for size,r in (((.02,.02,-.02),np.eye(3)),((.02,.02,.02),np.diag([1.,1.,-1.])),
                       ((.02,.02,.02),np.full((3,3),np.nan))):
            with self.assertRaises(ValueError):ws.check_oriented_box((0,0,.95),size,r)

    def supported_payload(self,ws):
        ref='camera/observed/payload';ws.scene.masks={ref:np.ones((50,50),bool)}
        proof=dict(time_s=1.,mask_refs=[ref],surface_inlier_ratio=1.,top_coverage=.5,
                   side_points=25,plane_residual_m=.0001,surface_tilt_deg=0.)
        self.assertTrue(ws.confirm_payload('visual_held',(0,0,.95),np.eye(3),(.04,.02,.02),proof))

    def test_supported_payload_replaces_only_its_clipped_proxy_not_other_objects(self):
        ws=self.workspace(valid=False)
        ws.objects=(AABB('visual_held',(0,0,.95),(.15,.15,.15)),
                    AABB('other_visual',(-.04,0,.95),(.01,.01,.01)))
        ws.build();self.assertIsNotNone(ws.unknown)
        self.supported_payload(ws);self.assertIsNone(ws.unknown)
        np.testing.assert_array_equal(ws._modelled(np.array([[0,0,.95],[.04,0,.95],[-.04,0,.95]])),
                                      [True,False,True])
        self.assertEqual(ws.check_oriented_box((.04,0,.95),(.005,.005,.005),np.eye(3)),
                         'UNOBSERVED_PATH_SPACE')

    def test_stale_payload_geometry_is_not_modelled_as_current(self):
        ws=self.workspace(valid=False);self.supported_payload(ws)
        self.assertTrue(ws._modelled(np.array([[0,0,.95]]))[0])
        ws.packet.assembled_time_s=1.2
        self.assertFalse(ws._modelled(np.array([[0,0,.95]]))[0])

    def test_current_payload_retains_visual_uncertainty_and_mapping_allowance_not_new_sweep(self):
        ws=self.workspace(valid=False);self.supported_payload(ws)
        self.assertEqual(ws.payload_observation['uncertainty_m'],.003)
        self.assertIsNone(ws.check_oriented_box((0,0,.95),(.050,.030,.030),np.eye(3)))
        self.assertEqual(ws.check_oriented_box((.001,0,.95),(.050,.030,.030),np.eye(3)),
                         'UNOBSERVED_PATH_SPACE')

    def test_invalid_payload_uncertainty_revokes_previous_support(self):
        ws=self.workspace(valid=False)
        ref='camera/observed/payload';ws.scene.masks={ref:np.ones((50,50),bool)}
        proof=dict(time_s=1.,mask_refs=[ref],surface_inlier_ratio=1.,top_coverage=.5,
                   side_points=25,plane_residual_m=.0001,surface_tilt_deg=0.)
        for value in (float('nan'),float('inf'),.002,.009):
            self.supported_payload(ws)
            self.assertFalse(ws.confirm_payload('visual_held',(0,0,.95),np.eye(3),(.04,.02,.02),proof,value))
            self.assertIsNone(ws.payload_observation)
