import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from workstation.diagnostics.graspgen_pair import drive_empty_pregrasps, paired_pregrasp, synchronized_knots
from workstation.planning.selection import CandidateAssessment, GraspCandidate
from workstation.simulation.visual_fixtures import initial_cartons


class PairTests(unittest.TestCase):
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
        result=drive_empty_pregrasps(robot,paths)
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
