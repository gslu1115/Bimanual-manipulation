import json
import unittest
from workstation.app import build_parser,load_config,PROJECT_ROOT


class VisualCLITests(unittest.TestCase):
    def test_calibrated_sim_depth_margin_is_explicit_with_static_board(self):
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp',
            '--depth-backboard','--optical-depth-margin-mm','1']),parser)
        from workstation.perception.scene_estimate import ScenePriors
        self.assertEqual(ScenePriors.from_config(config).optical_depth_error_margin_m,.001)

    def test_low_margin_without_calibration_fixture_is_rejected(self):
        import contextlib,io
        parser=build_parser()
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            load_config(parser.parse_args(['--mode','visual-pnp','--optical-depth-margin-mm','1']),parser)

    def test_lower_scene_installation_reuses_same_calibration_pipeline(self):
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp',
            '--scene-view','west','--scene-height-m','1.3']),parser)
        scene=next(c for c in config['cameras'] if c['name']=='scene_camera')
        self.assertEqual(scene['position'],[-.85,-.10,1.3]);self.assertEqual(len(config['cameras']),3)

    def test_wider_wrists_are_explicit_and_preserve_camera_system(self):
        path=PROJECT_ROOT/'config/scene.json';original=path.read_bytes()
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp',
            '--wrist-focal-length-mm','12']),parser)
        self.assertEqual(len(config['cameras']),3)
        self.assertEqual([c['focal_length_mm'] for c in config['cameras']], [36.,12.,12.])
        self.assertEqual(path.read_bytes(),original)

    def test_observer_preparation_remains_an_explicit_skill_option(self):
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp','--preposition-observer']),parser)
        self.assertTrue(config['diagnostic_preposition_observer'])
        self.assertEqual(len(config['cameras']),3)

    def test_stow_home_is_only_an_explicit_initial_condition(self):
        path=PROJECT_ROOT/'config/scene.json';original=path.read_bytes()
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp','--robot-stow-home']),parser)
        self.assertEqual(config['robot_home'],[0.,-.55,0.,-1.9,0.,1.35,.7853981633974483,.04,.04])
        self.assertTrue(config['diagnostic_robot_stow_home'])
        self.assertEqual(path.read_bytes(),original)

    def test_opposed_side_view_keeps_original_camera_system(self):
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp','--scene-view','west']),parser)
        scene=next(c for c in config['cameras'] if c['name']=='scene_camera')
        self.assertEqual(scene['position'],[-.85,-.10,1.7]);self.assertEqual(len(config['cameras']),3)

    def test_overhead_view_is_explicit_and_preserves_source_config(self):
        original=(PROJECT_ROOT/'config/scene.json').read_bytes()
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp',
            '--scene-view','overhead','--scene-look-height-m','1.05']),parser)
        scene=next(c for c in config['cameras'] if c['name']=='scene_camera')
        self.assertEqual(scene['position'],[0.,.04,2.1]);self.assertEqual(scene['look_at'][2],1.05)
        self.assertEqual(len(config['cameras']),3)
        self.assertEqual((PROJECT_ROOT/'config/scene.json').read_bytes(),original)

    def test_default_camera_config_and_file_preserved(self):
        path=PROJECT_ROOT/'config/scene.json';original=path.read_bytes()
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp']),parser)
        self.assertEqual(config['cameras'],json.loads(original)['cameras'])
        self.assertEqual(path.read_bytes(),original)

    def test_viewpoint_experiments_keep_three_cameras_and_recalibrate_params(self):
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp',
            '--scene-focal-length-mm','20','--scene-look-height-m','1.05','--right-wrist-view','cross-left']),parser)
        self.assertEqual({c['name'] for c in config['cameras']},
                         {'scene_camera','left_wrist_camera','right_wrist_camera'})
        scene=next(c for c in config['cameras'] if c['name']=='scene_camera')
        self.assertEqual(scene['focal_length_mm'],20);self.assertEqual(scene['look_at'][2],1.05)
        self.assertEqual(config['diagnostic_right_wrist_view'],'cross-left')

    def test_explicit_separated_task_and_item_bound(self):
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-sort',
            '--visual-fixture','separated-upright','--max-items','3']),parser)
        self.assertEqual(config['boxes']['count'],3)
        self.assertEqual(config['diagnostic_visual_fixture'],'separated-upright')

    def test_side_bracket_is_an_explicit_initial_mount_only(self):
        parser=build_parser();config=load_config(parser.parse_args(['--mode','visual-pnp',
            '--right-wrist-view','cross-left-offset']),parser)
        right=next(c for c in config['cameras'] if c['name']=='right_wrist_camera')
        self.assertEqual(right['mount_translation'],[.065,.060,.025])
        self.assertEqual(right['type'],'wrist');self.assertEqual(len(config['cameras']),3)
