import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tests import test_visual_planning
from workstation.planning.graspgen_candidates import candidates_from_result,revalidate_candidate,upright_body_rotation,finger_preload_target
from workstation.models.graspgen_source import GraspGenSource
from workstation.skills.visual_pick_place import VisualPickPlace
from workstation.observations.observation_packet import RobotState


class GraspGenExecutionTests(unittest.TestCase):
    def test_preload_depends_on_current_fingers_and_keeps_command_limits(self):
        self.assertAlmostEqual(finger_preload_target([.0277,.0278],.054),.0237)
        self.assertAlmostEqual(finger_preload_target([.032,.031],.060),.027)
        self.assertEqual(finger_preload_target([.002,.002],.004),0.)
        # An existing smaller closing command must not be opened by preload.
        self.assertEqual(finger_preload_target([.0277,.0278],.040),.020)

    def test_preload_rejects_bad_feedback_or_unbounded_compression(self):
        for fingers,width,amount in (([float('nan'),.02],.04,.004),([-.01,.02],.04,.004),
                                     ([.02,.02],.09,.004),([.02,.02],.04,.02)):
            with self.assertRaises(ValueError):finger_preload_target(fingers,width,amount)

    def setUp(self):
        fixture=test_visual_planning.PlanningTests();fixture.setUp()
        self.scene,self.states=fixture.scene,fixture.states;self.obj=self.scene.objects[0]
        self.calibration={'verified':True,'T_graspgen_base_from_project_tcp':np.eye(4).tolist()}
        pose=np.eye(4);pose[:3,:3]=upright_body_rotation(self.obj)@np.diag([1.,-1.,-1.])
        pose[:3,3]=np.asarray(self.obj.position_m)+[0,0,.009]
        self.result=dict(base_poses_workcell=np.array([pose]),scores=np.array([.8]),request_id='original',
            observation_time_s=self.obj.time_s,metadata={'gripper_name':'franka_panda'},round_trip_s=.2)
        self.candidate=candidates_from_result(self.result,self.obj,self.states,self.calibration,{'cam':1})[0]

    def fresh(self,delta=None):
        obj=replace(self.obj,time_s=self.obj.time_s+.3,
            position_m=tuple(np.asarray(self.obj.position_m)+(np.zeros(3) if delta is None else delta)))
        scene=replace(self.scene,objects=(obj,),observation_time_s=obj.time_s,assembled_time_s=obj.time_s)
        return obj,scene

    def test_new_observation_retains_original_model_pose_and_source_time(self):
        obj,scene=self.fresh([.0005,0,0])
        current=revalidate_candidate(self.candidate,obj,scene,{'cam':8})
        self.assertEqual(current.position_m,self.candidate.position_m)
        self.assertEqual(current.orientation_wxyz,self.candidate.orientation_wxyz)
        self.assertEqual(current.observation_time_s,self.candidate.observation_time_s)
        self.assertEqual(current.validated_time_s,obj.time_s)
        self.assertEqual(current.metadata['revalidation']['sequence_ids'],{'cam':8})

    def test_moved_ambiguous_or_old_target_cannot_reuse_world_pose(self):
        obj,scene=self.fresh([.003,0,0])
        with self.assertRaisesRegex(RuntimeError,'MODEL_TARGET_CHANGED'):
            revalidate_candidate(self.candidate,obj,scene,{'cam':8})
        obj,scene=self.fresh()
        with self.assertRaisesRegex(RuntimeError,'MODEL_TARGET_UNCERTAIN'):
            revalidate_candidate(self.candidate,replace(obj,track_id='other'),scene,{'cam':8})
        with self.assertRaisesRegex(RuntimeError,'MODEL_TARGET_STALE'):
            revalidate_candidate(self.candidate,self.obj,self.scene,{'cam':1})
        with self.assertRaisesRegex(RuntimeError,'MODEL_TARGET_STALE'):
            revalidate_candidate(self.candidate,obj,replace(scene,assembled_time_s=obj.time_s+.2),{'cam':8})

    def test_skill_refresh_does_not_restore_upright_template_position(self):
        skill=VisualPickPlace.__new__(VisualPickPlace)
        obj,skill.scene=self.fresh()
        skill.packet=SimpleNamespace(cameras={'cam':SimpleNamespace(sequence_id=8)})
        refreshed=skill._refresh_candidate(self.candidate,obj)
        self.assertEqual(refreshed.position_m,self.candidate.position_m)
        self.assertNotAlmostEqual(refreshed.position_m[2],obj.position_m[2]+.004)
        self.assertEqual(refreshed.candidate_id,self.candidate.candidate_id)

    def test_online_request_advances_physics_and_uses_later_frame(self):
        class Pending:
            checks=0
            def done(inner):inner.checks+=1;return inner.checks>1
            def result(inner):return self.result
        class Pool:
            def __init__(inner,**kw):pass
            def __enter__(inner):return inner
            def __exit__(inner,*args):pass
            def submit(inner,*args):return Pending()
        time_s=[self.obj.time_s]
        robot=SimpleNamespace(hold=lambda seconds:time_s.__setitem__(0,time_s[0]+seconds),
            read_states=lambda:{name:RobotState(time_s[0],state.joint_positions_rad,state.finger_positions_m)
                                for name,state in self.states.items()})
        def refresh():
            obj=replace(self.obj,time_s=time_s[0])
            return SimpleNamespace(cameras={'cam':SimpleNamespace(sequence_id=8)}),replace(self.scene,
                objects=(obj,),observation_time_s=time_s[0],assembled_time_s=time_s[0])
        source=GraspGenSource(SimpleNamespace(infer=None),self.calibration)
        with patch('workstation.models.graspgen_source.ThreadPoolExecutor',Pool),patch(
                'workstation.models.graspgen_source.visual_clouds',return_value=SimpleNamespace(time_s=self.obj.time_s)):
            rows=source.generate(SimpleNamespace(cameras={'cam':SimpleNamespace(sequence_id=1)}),self.scene,robot,refresh)
        self.assertEqual(len(rows),2);self.assertTrue(all(not x.rejection_reasons for x in rows))
        self.assertGreater(source.requests[0]['simulation_advanced_s'],0)
        self.assertGreater(rows[0].validated_time_s,rows[0].observation_time_s)

    def test_model_hold_observer_runs_before_pose_evidence_expires(self):
        skill=VisualPickPlace.__new__(VisualPickPlace)
        skill._monitor_ticks=0;skill.grasp_source=object();skill.held=False
        skill.priors=SimpleNamespace(max_age_s=.15);skill.robot=SimpleNamespace(dt=1/120)
        calls=[];skill.observe=lambda:calls.append(skill._monitor_ticks)
        for _ in range(30):skill.monitor()
        self.assertGreater(len(calls),2)
        self.assertLess(max(np.diff([0]+calls))*skill.robot.dt,skill.priors.max_age_s)
