"""Three public size priors, measured selection, and one centre-placement goal."""
import contextlib
import io
import json
import math
from pathlib import Path
from dataclasses import replace
import unittest
import numpy as np
from tests.test_scene_estimate import packet
from workstation.app import build_parser,load_config
from workstation.perception.scene_estimate import ScenePriors
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.planning.task_targets import visual_target_slots
from workstation.simulation.visual_fixtures import initial_cartons
from workstation.skills.cumotion_pick_place import placement_goal


class SmallCartonCentreTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text())
        self.priors=ScenePriors.from_config(self.config)
        self.sizes=self.priors.carton_size_catalog_m

    def test_each_size_is_selected_from_rgbd_at_different_angles(self):
        for size in self.sizes:
            for angle in (0,37,88,135,233,317):
                with self.subTest(size=size,angle=angle):
                    scene=CartonEstimator(self.priors).estimate(packet(size=size,yaw=math.radians(angle)))
                    self.assertEqual(len(scene.objects),1)
                    obj=scene.objects[0]
                    self.assertEqual(obj.size_m,size)
                    self.assertFalse(obj.failures)
                    self.assertEqual(obj.posture,'UPRIGHT')
                    np.testing.assert_allclose(obj.position_m,[0.,-.1,.76+size[2]/2],atol=.003)

    def test_scene_variant_changes_body_but_not_online_catalogue_or_cameras(self):
        parser=build_parser();positions=[]
        for name in ('small_reference','small_medium','small_wide'):
            c=load_config(parser.parse_args(['--mode','visual-pnp','--visual-fixture','upright',
                '--visual-carton-variant',name,'--seed','413','--visual-yaw-deg','88']),parser)
            self.assertEqual(ScenePriors.from_config(c),self.priors)
            self.assertEqual(c['cameras'],self.config['cameras'])
            box=initial_cartons(c,'upright')[0];positions.append(box.position[:2])
            self.assertEqual(box.size_m,next(v['size_m'] for v in c['boxes']['variants'] if v['name']==name))
            self.assertAlmostEqual(box.position[2],.76+box.size_m[2]/2+.004)
        self.assertEqual(positions,[positions[0]]*3)

    def test_variant_does_not_unlock_large_or_other_postures(self):
        parser=build_parser()
        for options in (['--mode','visual-pnp','--visual-fixture','upright','--visual-carton-variant','large_a'],
                        ['--mode','visual-pnp','--visual-fixture','side','--visual-carton-variant','small_wide']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                load_config(parser.parse_args(options),parser)

    def test_only_one_visual_target_even_when_three_legacy_slots_exist(self):
        self.assertEqual(visual_target_slots(self.config),((-.2,-.4),))
        for index in (True,-1,3,[0,1,2]):
            c={**self.config,'visual_task':{'target_slot_index':index}}
            with self.assertRaises(ValueError):visual_target_slots(c)

    def test_grasp_offset_compensation_places_each_body_center_at_same_xy(self):
        for size in self.sizes:
            for angle in (0,88,233):
                a=math.radians(angle);c,s=math.cos(a),math.sin(a)
                body=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                tcp=np.diag([1.,-1.,-1.]);offset=np.array([.013,-.008,.006])
                p,r=placement_goal((-.2,-.4),.8,size,.01,body,offset,tcp.T@body)
                np.testing.assert_allclose(p+r@offset,[-.2,-.4,.8+size[2]/2+.01],atol=1e-12)

    def test_clipped_view_cannot_shrink_established_carton(self):
        est=CartonEstimator(self.priors)
        wide=self.sizes[-1];small=self.sizes[0]
        first=est.estimate(packet(size=wide)).objects[0]
        p=packet(time=1.2,size=wide);f=p.cameras['scene_camera'];rgb=f.rgb.copy();depth=f.depth_m.copy()
        template=packet(time=1.2,size=small).cameras['scene_camera']
        # Occlude the outer part without changing the remaining wide-box depth.
        hidden=(template.rgb[...,0]==100)&(rgb[...,0]!=100)
        rgb[hidden]=100;depth[hidden]=1.5-.76
        scene=est.estimate(replace(p,cameras={**p.cameras,'scene_camera':replace(f,rgb=rgb,depth_m=depth)}))
        self.assertTrue(scene.objects)
        obj=next(o for o in scene.objects if o.track_id==first.track_id)
        self.assertEqual(obj.size_m,wide)
        self.assertTrue(obj.failures or obj.visibility=='OBSERVED_CUBOID')

    def test_bad_catalogue_fails_before_inference(self):
        for catalogue in (((.1,.05),),((.1,-.05,.04),),((.1,float('nan'),.04),)):
            with self.assertRaises(ValueError):replace(self.priors,carton_size_catalog_m=catalogue)
