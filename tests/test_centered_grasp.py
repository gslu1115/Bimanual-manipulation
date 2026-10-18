import contextlib
import io
import math
import unittest
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from tests.test_scene_estimate import packet
from workstation.app import build_parser,load_config
from workstation.perception.scene_estimate import ScenePriors
from workstation.perception.rgbd_cartons import CartonEstimator
from workstation.planning.centered_grasp import centered_candidates
from workstation.planning.graspgen_candidates import revalidate_candidate
from workstation.observations.camera_geometry import pose_matrix


class CenteredGraspTests(unittest.TestCase):
    def test_three_sizes_and_yaws_have_central_symmetric_opening_and_top_clearance(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text())
        priors=ScenePriors.from_config(config)
        for size in priors.carton_size_catalog_m:
            for yaw in (0,29,88,108,127,209,317):
                obj=CartonEstimator(priors).estimate(packet(size=size,yaw=math.radians(yaw))).objects[0]
                rows=centered_candidates(obj,('left','right'),{})
                self.assertEqual(len(rows),4)
                for row in rows:
                    self.assertFalse(row.rejection_reasons)
                    self.assertLessEqual(row.opening_m,.08)
                    self.assertIsNone(row.model_score)
                    self.assertEqual(row.source,'centered')
                    np.testing.assert_allclose(row.position_m[:2],obj.position_m[:2])
                    self.assertAlmostEqual(row.position_m[2],obj.position_m[2]+size[2]/2-.012)
                    np.testing.assert_allclose(pose_matrix(row.position_m,row.orientation_wxyz)[:3,2],[0,0,-1],atol=1e-12)
                    self.assertAlmostEqual(row.metadata['opening']['centre_offset_m'],0)

    def test_centered_is_explicit_single_upright_controller_mode_without_model_assets(self):
        parser=build_parser()
        config=load_config(parser.parse_args(['--mode','visual-pnp','--grasp-backend','centered',
                                             '--motion-backend','cumotion']),parser)
        self.assertEqual(config['grasp_backend'],'centered')
        for args in (['--mode','visual-sort'],['--mode','visual-pnp'],['--mode','visual-adjust','--motion-backend','cumotion']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                load_config(parser.parse_args(args+['--grasp-backend','centered']),parser)

    def test_refresh_keeps_original_pose_and_rejects_moved_box(self):
        priors=ScenePriors.from_config(json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text()))
        est=CartonEstimator(priors);first=est.estimate(packet(time=1.))
        candidate=centered_candidates(first.objects[0],('left',),{})[0]
        scene=est.estimate(packet(time=1.1));obj=scene.objects[0]
        fresh=revalidate_candidate(candidate,obj,scene,{})
        self.assertEqual(fresh.position_m,candidate.position_m)
        self.assertEqual(fresh.observation_time_s,candidate.observation_time_s)
        moved=replace(obj,position_m=(obj.position_m[0]+.02,*obj.position_m[1:]))
        with self.assertRaisesRegex(RuntimeError,'MODEL_TARGET_CHANGED'):
            revalidate_candidate(candidate,moved,scene,{})
