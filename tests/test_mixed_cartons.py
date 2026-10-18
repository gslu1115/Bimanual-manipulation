"""Mixed-size construction/initialization; no claim of GPU settling or grasp success."""
import copy
import contextlib
import io
import json
import math
from pathlib import Path
import unittest

from workstation.app import build_parser,load_config
from workstation.simulation.cartons import carton_specs,visual_parts,box_size
from workstation.task_logic import BoxState,release_states,rotation,slot_errors,validate


class MixedCartonTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text())

    def test_recipe_has_six_graspable_and_three_all_axis_oversize_boxes(self):
        specs=carton_specs(self.config)
        self.assertEqual(len(specs),9)
        self.assertEqual(sum(s.group=='small' for s in specs),6)
        self.assertEqual(sum(s.group=='large' for s in specs),3)
        self.assertEqual(len({s.size_m for s in specs}),6)
        for spec in specs:
            if spec.group=='large':self.assertGreater(min(spec.size_m),.08)
            else:
                self.assertLessEqual(spec.size_m[1],.065)
                self.assertTrue(all(a>=b for a,b in zip(spec.size_m,self.config['boxes']['size'])))
            self.assertGreaterEqual(spec.mass_kg,.08-1e-12)

    def test_packaging_does_not_encode_size_class_and_follows_each_body(self):
        expected=None
        for spec in carton_specs(self.config):
            parts=visual_parts(spec.size_m);style=[(p[0],p[3],p[4]) for p in parts]
            if expected is None:expected=style
            self.assertEqual(style,expected)
            self.assertEqual(parts[0][2],spec.size_m)
            self.assertEqual([p[0] for p in parts if p[4]],['Cardboard'])
            self.assertAlmostEqual(parts[1][2][0],.97*spec.size_m[0])
            self.assertEqual(parts[1][2][1],.012)
            self.assertEqual(parts[2][2][1],.0007)
            self.assertGreater(parts[2][1][2],parts[1][1][2])
            self.assertEqual(parts[3][1][0],-spec.size_m[0]/2-.0001)
            self.assertEqual(parts[4][1][0],spec.size_m[0]/2+.0001)

    def test_seeded_release_does_not_intersect_and_is_inside_tray(self):
        for seed in range(60):
            boxes=release_states(self.config,seed);bounds=[]
            for box in boxes:
                r=rotation(box.orientation_wxyz)
                half=[sum(abs(r[i][j])*box.size_m[j]/2 for j in range(3)) for i in range(3)]
                for axis in (0,1):
                    self.assertLess(abs(box.position[axis]-self.config['upstream']['center_xy'][axis])+half[axis],
                                    self.config['upstream']['tray_size_xy'][axis]/2-.006)
                self.assertGreater(box.position[2]-half[2],self.config['table']['top_z']+.026)
                self.assertLess(box.position[2]+half[2],1.5)  # low release layers, not a nine-body tower
                bounds.append((box.position,half))
            for index,(p,h) in enumerate(bounds):
                for other,g in bounds[index+1:]:
                    self.assertTrue(any(abs(p[j]-other[j])>=h[j]+g[j]+.0119 for j in range(3)),(seed,p,other))
        self.assertEqual(release_states(self.config,413),release_states(self.config,413))
        self.assertNotEqual(release_states(self.config,413),release_states(self.config,414))

    def test_scene_defaults_to_mixed_but_controlled_fixtures_remain_uniform(self):
        parser=build_parser();scene=load_config(parser.parse_args([]),parser)
        self.assertEqual(scene['boxes']['size_set'],'mixed')
        self.assertEqual(scene['cameras'],self.config['cameras'])
        self.assertEqual(scene['robots'],self.config['robots'])
        for spec in carton_specs(scene):
            self.assertTrue(all(spec.size_m[j]<scene['conveyor']['slot_size_xy'][j] for j in (0,1)))
        for mode in ('single-pnp','dual-handover','vision-check','visual-pnp','sensor-check','model-capture'):
            c=load_config(parser.parse_args(['--mode',mode]),parser)
            self.assertEqual(c['boxes']['size_set'],'uniform')
            self.assertTrue(all(box.size_m is None for box in release_states(c,6)))
            self.assertEqual(c['conveyor'],self.config['conveyor'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            load_config(parser.parse_args(['--mode','visual-pnp','--box-set','mixed']),parser)

    def test_uniform_scene_preserves_original_release_and_record_schema(self):
        parser=build_parser();c=load_config(parser.parse_args(['--box-set','uniform']),parser)
        old=copy.deepcopy(c);old['boxes'].pop('size_set');old['boxes'].pop('variants')
        self.assertEqual(release_states(c,6),release_states(old,6))
        self.assertEqual(set(release_states(c,6)[0].to_dict()),
            {'name','position','orientation_wxyz','linear_velocity','angular_velocity'})

    def test_ground_truth_slot_height_uses_each_boxes_actual_size(self):
        for spec in carton_specs(self.config):
            box=BoxState('test',[-.2,-.4,self.config['conveyor']['top_z']+spec.size_m[2]/2],
                [1,0,0,0],[0,0,0],[0,0,0],list(spec.size_m),spec.mass_kg,spec.variant)
            self.assertEqual(slot_errors(box,[-.2,-.4],self.config)['z_m'],0.)
            self.assertEqual(box_size(box,self.config),list(spec.size_m))
            self.assertEqual(box.to_dict()['size_m'],list(spec.size_m))

    def test_invalid_size_classes_and_nonfinite_geometry_are_rejected(self):
        for dims in ([.13,.1,.08],[.13,.07,.09],[.13,.1],[.13,.1,float('nan')]):
            c=copy.deepcopy(self.config);c['boxes']['variants'][-1]['size_m']=dims
            with self.subTest(dims=dims),self.assertRaises(ValueError):validate(c)
        c=copy.deepcopy(self.config);c['boxes']['variants'][0]['size_m']=[.1,.04,.03]
        with self.assertRaises(ValueError):validate(c)


if __name__=='__main__':unittest.main()
