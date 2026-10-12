import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
from tests import test_visual_planning
from workstation.planning.graspgen_candidates import candidates_from_result,revalidate_candidate,upright_body_rotation,finger_preload_target
from workstation.models.graspgen_source import GraspGenSource
from workstation.skills.visual_pick_place import VisualPickPlace
from workstation.observations.observation_packet import RobotState


class GraspGenExecutionTests(unittest.TestCase):
    def test_normal_stage_anchor_uses_current_sample_and_requires_payload_evidence(self):
        skill=VisualPickPlace.__new__(VisualPickPlace);calls=[]
        skill.robot=SimpleNamespace(anchor_observation=lambda:calls.append(True))
        skill.packet=SimpleNamespace(observation_time_s=8.);skill.observation_anchors=[];skill.held=False
        skill._anchor_normal_stage('pregrasp',None)
        self.assertEqual(len(calls),1);self.assertEqual(skill.observation_anchors[-1]['time_s'],8.)
        skill.held=True;skill.packet.observation_time_s=23.
        skill._anchor_normal_stage('lifted',None);self.assertEqual(len(calls),1)
        skill._anchor_normal_stage('lifted',{'geometric_pose_measured':True})
        self.assertEqual(len(calls),2);self.assertEqual(skill.observation_anchors[-1]['time_s'],23.)
        skill._anchor_normal_stage('transferred',{'geometric_pose_measured':True})
        self.assertEqual(len(calls),2)

    def test_stage_observation_refresh_does_not_change_history_limits(self):
        from workstation.planning.occlusion_memory import OcclusionMemory
        memory=OcclusionMemory()
        def workspace(time):
            return SimpleNamespace(packet=SimpleNamespace(observation_time_s=time),_frames=lambda:[object()])
        old,current=workspace(8.),workspace(23.)
        memory.anchor(old);memory.anchor(current)
        memory.add(workspace(39.))
        self.assertNotIn(old,memory.anchors)
        self.assertEqual(memory.anchors,[current])
        self.assertEqual(old.packet.observation_time_s,8.)

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

    def generate_with_fresh_changes(self,changes):
        clock=[self.obj.time_s];observations=[];holds=[]
        def hold(seconds):holds.append(seconds);clock[0]+=seconds
        robot=SimpleNamespace(hold=hold,read_states=lambda:{name:
            RobotState(clock[0],state.joint_positions_rad,state.finger_positions_m)
            for name,state in self.states.items()})
        def refresh():
            change=dict(changes[len(observations)]);age=change.pop('assembled_age_s',0.)
            obj=replace(self.obj,time_s=clock[0],**change);observations.append(obj)
            return SimpleNamespace(cameras={'cam':SimpleNamespace(sequence_id=10+len(observations))}),replace(
                self.scene,objects=(obj,),observation_time_s=obj.time_s,assembled_time_s=obj.time_s+age)
        pool=MagicMock()
        pool.__enter__.return_value.submit.return_value=SimpleNamespace(done=lambda:True,result=lambda:self.result)
        source=GraspGenSource(SimpleNamespace(infer=None),self.calibration)
        with patch('workstation.models.graspgen_source.ThreadPoolExecutor',return_value=pool),patch(
                'workstation.models.graspgen_source.visual_clouds',return_value=SimpleNamespace(time_s=self.obj.time_s)):
            rows=source.generate(SimpleNamespace(cameras={'cam':SimpleNamespace(sequence_id=1)}),self.scene,robot,refresh)
        self.assertEqual(pool.__enter__.return_value.submit.call_count,1)
        self.assertEqual(len(source.requests),1)
        return source,rows,observations,holds

    def test_transient_packaging_ambiguity_uses_new_positive_frame_and_original_pose(self):
        source,rows,frames,holds=self.generate_with_fresh_changes([{'posture':'UNKNOWN'},{}])
        self.assertEqual(len(frames),2);self.assertEqual(holds,[.025,.025])
        self.assertTrue(all(not row.rejection_reasons for row in rows))
        for row in rows:
            self.assertEqual(row.position_m,self.candidate.position_m)
            self.assertEqual(row.orientation_wxyz,self.candidate.orientation_wxyz)
            self.assertEqual(row.observation_time_s,self.obj.time_s)
            self.assertEqual(row.validated_time_s,frames[-1].time_s)
        self.assertEqual(source.requests[0]['additional_passive_hold_s'],.025)

    def test_persistent_unknown_stops_after_two_extra_samples_without_old_label(self):
        source,rows,frames,holds=self.generate_with_fresh_changes([{'posture':'UNKNOWN'}]*3)
        self.assertEqual(len(frames),3);self.assertEqual(holds,[.025]*3)
        self.assertTrue(all(row.rejection_reasons==('MODEL_TARGET_UNCERTAIN',) for row in rows))
        self.assertTrue(all(row.validated_time_s is None for row in rows))
        self.assertEqual(source.requests[0]['additional_passive_hold_s'],.05)

    def test_later_positive_semantics_cannot_rescue_changed_geometry(self):
        _,rows,frames,_=self.generate_with_fresh_changes([{'posture':'UNKNOWN'},
            {'position_m':tuple(np.asarray(self.obj.position_m)+[.01,0,0])}])
        self.assertEqual(len(frames),2)
        self.assertTrue(all(row.rejection_reasons==('MODEL_TARGET_CHANGED',) for row in rows))

    def test_stale_unknown_frame_does_not_enter_passive_retry(self):
        _,rows,frames,_=self.generate_with_fresh_changes([{'posture':'UNKNOWN','assembled_age_s':.2}])
        self.assertEqual(len(frames),1)
        self.assertTrue(all(row.rejection_reasons for row in rows))

    def test_partial_geometry_or_positive_inversion_is_not_transient_packaging(self):
        for change in ({'posture':'UNKNOWN','failures':('PARTIAL_OR_MERGED_GEOMETRY',)},
                       {'posture':'INVERTED'}):
            with self.subTest(change=change):
                _,rows,frames,_=self.generate_with_fresh_changes([change])
                self.assertEqual(len(frames),1)
                self.assertTrue(all(row.rejection_reasons for row in rows))
