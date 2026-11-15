import unittest
from types import SimpleNamespace as NS
import numpy as np
from workstation.planning.collision import AABB,scene_obstacles,check_spheres,carton_contact_allowed
from workstation.simulation.robot_driver import RobotMotion


class CartonContactTests(unittest.TestCase):
    def fixture(self):
        # Current depth spans from the failed four-carton run, not simulator poses.
        obj=NS(track_id='wide',time_s=1.,position_m=(0.,0.,1.),axis_yaw_rad=np.pi/4,
            size_m=(.12,.065,.055),uncertainty_m=.003,visibility='PARTIAL_OR_UNCERTAIN',
            failures=('PARTIAL_OR_MERGED_GEOMETRY',),mask_refs=('scene_camera/1/a',),
            quality=dict(geometric_length_m=.12,geometric_width_m=.065,observed_height_m=.055,
                observed_length_m=.12000997,observed_width_m=.06498374,observed_vertical_span_m=.05246462,
                surface_tilt_deg=.233084,plane_residual_m=.00035066,geometry_inlier_ratio=1.,footprint_coverage=.68799))
        scene=NS(objects=[obj],frame='workcell',observation_time_s=1.,assembled_time_s=1.,
            masks={'scene_camera/1/a':True},camera_status={'scene_camera':'OK'})
        return obj,scene

    def test_clipped_area_preserves_supported_three_axis_extent(self):
        obj,scene=self.fixture();box=scene_obstacles(scene,None)[0]
        self.assertLess(box.size[0],.146);self.assertLess(box.size[2],.070)
        self.assertGreater(box.size[2],.055+.014)
        self.assertEqual(obj.failures,('PARTIAL_OR_MERGED_GEOMETRY',))
        self.assertEqual(obj.visibility,'PARTIAL_OR_UNCERTAIN')

    def test_incomplete_spans_conflicts_missing_masks_and_stale_frames_remain_conservative(self):
        for mutation in (lambda o,s:o.quality.update(observed_width_m=.040),
                         lambda o,s:o.quality.update(observed_vertical_span_m=.02),
                         lambda o,s:o.quality.update(surface_tilt_deg=12.),
                         lambda o,s:o.quality.update(geometry_inlier_ratio=.8),
                         lambda o,s:o.quality.update(observed_length_m=float('nan')),
                         lambda o,s:setattr(o,'failures',('PACKAGING_CUE_CONFLICT',)),
                         lambda o,s:setattr(o,'time_s',0.),
                         lambda o,s:setattr(s,'assembled_time_s',2.),
                         lambda o,s:s.masks.clear(),
                         lambda o,s:s.camera_status.update(scene_camera='MISSING')):
            obj,scene=self.fixture();mutation(obj,scene)
            self.assertAlmostEqual(scene_obstacles(scene,None)[0].size[2],.134)

    def test_contact_permission_requires_both_typed_cartons(self):
        a=AABB('a',(0,0,0),(1,1,1),'carton')
        self.assertTrue(carton_contact_allowed(a,a))
        for kind in ('fixed','unknown','robot'):
            self.assertFalse(carton_contact_allowed(a,AABB('b',(0,0,0),(1,1,1),kind)))

    def test_robot_self_mutual_fixed_and_carton_protection_unchanged(self):
        sphere=(4,np.zeros(3),.02)
        self.assertEqual(check_spheres([sphere],[(4,np.zeros(3),.02)],[],'mount')[0],'OTHER_ARM_COLLISION')
        self.assertEqual(check_spheres([sphere,(7,np.zeros(3),.02)],[],[],'mount')[0],'SELF_COLLISION')
        for kind in ('fixed','carton'):
            detail={}
            result=check_spheres([sphere],[],[AABB('body',(0,0,0),(.1,.1,.1),kind)],'mount',rejection=detail)
            self.assertEqual(result[0],'STATIC_PATH_BLOCKED:body')
            self.assertEqual(detail['link_index'],4);self.assertEqual(detail['obstacle_kind'],kind)

    def test_actual_held_path_allows_carton_contact_but_not_fixed_collision(self):
        robot=RobotMotion.__new__(RobotMotion)
        robot.read_states=lambda:{a:NS(joint_positions_rad=np.zeros(7),finger_positions_m=np.full(2,.02)) for a in ('left','right')}
        robot.spheres=lambda *args:[]
        robot.fk=lambda *args:(np.array([0.,0.,1.]),np.eye(3))
        robot.limits=(np.full(7,-10.),np.full(7,10.))
        robot.static=[]
        # Stationary held/parked neighbour follows the same path policy.
        for kind in ('carton','fixed'):
            robot.stationary_payload_keepout=AABB('neighbour',(0.,0.,1.),(.1,.1,.1),kind)
            def run():robot.validate_path('left',[np.zeros(7),np.full(7,.01)],NS(objects=[]),'held',(.1,.055,.045))
            if kind=='carton':run()
            else:
                with self.assertRaisesRegex(RuntimeError,'HELD_CARTON_PATH_BLOCKED:neighbour'):run()

    def test_allowed_contact_does_not_bypass_unknown_space(self):
        robot=RobotMotion.__new__(RobotMotion)
        robot.read_states=lambda:{a:NS(joint_positions_rad=np.zeros(7),finger_positions_m=np.full(2,.02)) for a in ('left','right')}
        robot.spheres=lambda *args:[];robot.static=[]
        robot.fk=lambda *args:(np.array([0.,0.,1.]),np.eye(3))
        robot.limits=(np.full(7,-10.),np.full(7,10.))
        robot.observed_workspace=NS(check_oriented_box=lambda *a,**k:'UNOBSERVED_PATH_SPACE',report=lambda:{})
        with self.assertRaisesRegex(RuntimeError,'UNOBSERVED_PATH_SPACE'):
            robot.validate_path('left',[np.zeros(7),np.full(7,.01)],NS(objects=[]),'held',(.1,.055,.045))
