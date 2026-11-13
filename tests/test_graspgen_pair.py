import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from workstation.diagnostics.graspgen_pair import drive_empty_pregrasps, paired_pregrasp, synchronized_knots
from workstation.planning.selection import CandidateAssessment, GraspCandidate
from workstation.simulation.visual_fixtures import initial_cartons


class PairTests(unittest.TestCase):
    def test_crossing_transfer_can_wait_and_rejects_blocked_endpoints(self):
        from workstation.diagnostics.graspgen_pair_place import transfer_schedule
        def spheres(arm,q):
            p=[q[0],0,0] if arm=='panda_left' else [0,q[0],0]
            return [(0,np.array(p),.025)]
        robot=SimpleNamespace(spheres=spheres)
        path=[np.full(7,x) for x in np.linspace(-.3,.3,21)]
        scheduled,detail=transfer_schedule(robot,[path,path],lambda l,r:None)
        self.assertGreater(detail['delay_fraction'],0.)
        for original,timed in zip((path,path),scheduled):
            np.testing.assert_array_equal(original[0],timed[0])
            np.testing.assert_array_equal(original[-1],timed[-1])
        blocked=path[:11]
        with self.assertRaisesRegex(RuntimeError,'NO_CLEAR'):
            transfer_schedule(robot,[blocked,blocked],lambda l,r:None)

    def test_resampling_uses_distinct_nearby_arms_and_actual_side_points(self):
        from workstation.diagnostics.graspgen_pair import preferred_assignments,balance_visible_surfaces
        from workstation.models.graspgen_client import VisualClouds
        def carton(name,x):
            return SimpleNamespace(track_id=name,position_m=[x,0,0],size_m=[.12,.06,.05],
                posture='UPRIGHT',visibility='COMPLETE_TOP',axis_yaw_rad=0.)
        a,b=carton('a',-.18),carton('b',.18)
        bases={'panda_left':{'base_xyz':[-.6,0,0]},'panda_right':{'base_xyz':[.6,0,0]}}
        self.assertEqual(preferred_assignments([b,a],bases),{'a':'panda_left','b':'panda_right'})
        rng=np.random.default_rng(4)
        top=rng.uniform([-.055,-.025,.025],[.055,.025,.025],(4000,3))
        side=rng.uniform([-.055,.03,-.020],[.055,.03,.020],(150,3))
        points=np.r_[top,side]+a.position_m
        clouds=VisualClouds(points,points,1.)
        chosen=balance_visible_surfaces(clouds,a)
        self.assertLess(len(chosen.target_m),len(points))
        self.assertGreater(np.mean(chosen.target_m[:,1]==.03),.30)
        self.assertTrue(set(map(tuple,chosen.target_m)).issubset(set(map(tuple,points))))
        self.assertIs(chosen.scene_m,points)

    def test_rotated_inference_preserves_observed_contact_in_world(self):
        from workstation.models.graspgen_client import VisualClouds, infer_rotated_view
        points=np.array([[.21,.15,.8],[.24,.19,.82],[.25,.10,.81]],np.float32)
        def predict(clouds,count):
            pose=np.eye(4)
            axis=clouds.target_m[1]-clouds.target_m[0]
            pose[:3,0]=axis/np.linalg.norm(axis)
            pose[:3,3]=clouds.target_m[0]
            return dict(base_poses_workcell=pose[None],metadata={})
        clouds=VisualClouds(points,points.copy(),.5)
        proxy=SimpleNamespace(infer=predict)
        expected=predict(clouds,1)['base_poses_workcell'][0]
        for pitch in (0.,np.pi/3,-np.pi/3):
            actual=infer_rotated_view(proxy,clouds,1,np.pi/4,pitch)['base_poses_workcell'][0]
            np.testing.assert_allclose(actual[:3,[0,3]],expected[:3,[0,3]],atol=2e-6)

    def test_model_is_default_for_single_and_mandatory_for_pair(self):
        import contextlib, io
        from workstation.app import build_parser, load_config
        parser=build_parser()
        for mode in ('visual-pnp','graspgen-pair-pregrasp'):
            args=parser.parse_args(['--mode',mode])
            self.assertEqual(load_config(args,parser)['grasp_backend'],'graspgen')
        args=parser.parse_args(['--mode','graspgen-pair-pregrasp','--grasp-backend','centered'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            load_config(args,parser)

    def test_pair_slot_subset_preserves_physical_conveyor_and_requires_distinct_lanes(self):
        import contextlib, io
        from workstation.app import build_parser,load_config
        parser=build_parser()
        args=parser.parse_args(['--mode','graspgen-pair-place','--pair-slots','0','1'])
        config=load_config(args,parser)
        self.assertEqual(config['pair_slot_indices'],[0,1])
        self.assertEqual(len(config['conveyor']['slots_xy']),3)
        args.pair_slots=[1,1]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            load_config(args,parser)

    def test_both_commands_share_one_physics_tick_and_overlap_uses_encoders(self):
        arms=('panda_left','panda_right')
        measured={a:np.zeros(7) for a in arms}
        commands={a:np.zeros(9) for a in arms}
        def tick():
            for a in arms:measured[a]=commands[a][:7].copy()
        robot=SimpleNamespace(dt=.02,joint_speed=.5,targets=commands,tick=tick,
            read_states=lambda:{a:SimpleNamespace(joint_positions_rad=q) for a,q in measured.items()},
            spheres=lambda a,q:[(0,np.array([q[0],float(a==arms[0]),0]),.02)])
        paths=[[np.zeros(7),np.full(7,.12)] for a in arms]
        elapsed=[]
        result=drive_empty_pregrasps(robot,paths,progress_check=elapsed.append)
        np.testing.assert_allclose(elapsed,np.arange(1,result['ticks']+1)*robot.dt)
        self.assertGreater(result['overlap_sim_s'],.15)
        for a in arms:
            measured[a]=np.zeros(7)
        robot.tick=lambda:None
        with self.assertRaisesRegex(RuntimeError,'TRACKING_ERROR'):
            drive_empty_pregrasps(robot,paths)

    def test_two_graspable_upright_cartons_at_distinct_starts(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text())
        for seed in range(6,12):
            cfg['seed']=seed
            a,b=initial_cartons(cfg,'upright-pair')
            self.assertNotEqual(a.variant,b.variant)
            for box in (a,b):
                self.assertEqual(box.orientation_wxyz[1:3],[0.,0.])
                self.assertLess(box.size_m[1],.08)
            self.assertGreater(b.position[0]-a.position[0],.34)

    def test_equal_size_fixture_preserves_distinct_instances_and_seeded_poses(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text())
        cfg['seed']=8022
        previous=initial_cartons(cfg,'upright-pair')
        cfg['diagnostic_pair_carton_variants']=['small_medium','small_medium']
        current=initial_cartons(cfg,'upright-pair')
        self.assertNotEqual(current[0].name,current[1].name)
        self.assertEqual(current[0].size_m,current[1].size_m)
        for a,b in zip(previous,current):
            self.assertEqual(a.position[:2],b.position[:2])
            self.assertEqual(a.orientation_wxyz,b.orientation_wxyz)
        cfg['diagnostic_pair_carton_variants']=['small_medium','missing']
        with self.assertRaises(ValueError):initial_cartons(cfg,'upright-pair')

    def test_parked_carton_is_physical_but_only_the_two_table_cartons_are_pick_targets(self):
        from workstation.diagnostics.graspgen_pair import source_cartons
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config/scene.json').read_text())
        cfg['seed']=8041
        for index,xy in enumerate(cfg['conveyor']['slots_xy']):
            cfg['diagnostic_occupied_slot']=index
            boxes=initial_cartons(cfg,'upright-pair')
            self.assertEqual(boxes[2].position[:2],xy)
            self.assertEqual(boxes[2].size_m,boxes[0].size_m)
            scene=SimpleNamespace(objects=[SimpleNamespace(track_id=b.name,position_m=b.position,
                posture='UPRIGHT',failures=(),visibility='COMPLETE_TOP') for b in boxes])
            self.assertEqual([o.track_id for o in source_cartons(scene,cfg['table'])],['box_00','box_01'])
            self.assertEqual(len(scene.objects),3)

    def test_parked_track_requires_a_single_supported_stationary_observation(self):
        from workstation.diagnostics.graspgen_pair_place import stationary_slot_tracks
        xy=(0.,-.4)
        o=SimpleNamespace(track_id='t',position_m=[*xy,.785],size_m=[.11,.06,.05],failures=(),
            posture='UPRIGHT',visibility='COMPLETE_TOP',velocity_m_s=[0.,0.,0.])
        ids=['t'];scene=SimpleNamespace(objects=[o])
        w=SimpleNamespace(slot=lambda _:dict(observed_objects=ids),priors=SimpleNamespace(support_z_m=.76))
        self.assertEqual(stationary_slot_tracks(w,scene,[xy]),{xy:o})
        ids.append('fragment')
        self.assertFalse(stationary_slot_tracks(w,scene,[xy]))
        ids.pop()
        o.velocity_m_s=[.02,0.,0.]
        self.assertFalse(stationary_slot_tracks(w,scene,[xy]))

    def test_both_paths_preserve_corners_and_limit_displacement(self):
        left=np.zeros((3,7)); left[1,0]=.21
        right=np.zeros((4,7)); right[:,0]=[0,.16,.08,.04]
        samples=np.array(list(synchronized_knots(left,right)))
        self.assertLessEqual(np.abs(np.diff(samples,axis=0)).max(),.025+1e-8)
        self.assertAlmostEqual(samples[:,0,0].max(),.21)
        self.assertAlmostEqual(samples[:,1,0].max(),.16)

    def test_simultaneous_crossing_and_duplicate_object_are_rejected(self):
        def candidate(arm,obj):
            c=GraspCandidate(obj,arm,(0,0,0),0,.07,0,source='graspgen',candidate_id=arm+obj)
            return CandidateAssessment(c,True,(),dict(full_pose_checked=True,pregrasp_joint_positions_rad=[1.]*7),1)
        arms=('panda_left','panda_right')
        robot=SimpleNamespace(checked_joint_plan=lambda a,start,goal,s,t:([start,goal],1),
            spheres=lambda a,q:[(0,np.array([q[0] if a==arms[0] else 1-q[0],0,0]),.02)])
        states={a:SimpleNamespace(joint_positions_rad=np.zeros(7)) for a in arms}
        a=candidate(arms[0],'one')
        self.assertEqual(paired_pregrasp(robot,[a,candidate(arms[1],'one')],states,None),(None,[]))
        pair,rejected=paired_pregrasp(robot,[a,candidate(arms[1],'two')],states,None)
        self.assertIsNone(pair)
        self.assertEqual(rejected[0]['reason'],'OTHER_ARM_COLLISION')
        robot.spheres=lambda a,q:[(0,np.array([q[0],float(a==arms[0]),0]),.02)]
        pair,_=paired_pregrasp(robot,[a,candidate(arms[1],'two')],states,None)
        self.assertEqual(len(pair['selection']),2)
        self.assertGreater(pair['min_clearance_m'],0)
