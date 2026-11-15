import math
import unittest
from types import SimpleNamespace as NS
import numpy as np
from workstation.planning.joint_pair_selection import (
    JointPairSelection,similar_grasp,visible_fraction,recoverable_empty_failure)


class JointPairTests(unittest.TestCase):
    def candidate(self,x=-.2,angle=0.,arm='panda_left',offset=0.):
        return NS(arm=arm,position_m=(x+offset,0.,.8),orientation_wxyz=(math.cos(angle/2),0.,0.,math.sin(angle/2)),
                  metadata={'source_visual_object':dict(position_m=(x,0.,.8),size_m=(.1,.055,.045))})

    def policy(self):
        return JointPairSelection({str(x):dict(position_m=(x,0.,.8),size_m=(.1,.055,.045)) for x in (-.2,0.,.2)})

    def test_pose_diversity_ignores_quaternion_sign_but_preserves_changed_wrist(self):
        a=vars(self.candidate());b=dict(a,orientation_wxyz=(-1.,0.,0.,0.))
        self.assertTrue(similar_grasp(a,b))
        self.assertTrue(similar_grasp(a,vars(self.candidate(offset=.003,angle=math.radians(10)))))
        self.assertFalse(similar_grasp(a,vars(self.candidate(angle=math.radians(25)))))
        self.assertFalse(similar_grasp(a,vars(self.candidate(offset=.009))))
        self.assertFalse(similar_grasp(a,vars(self.candidate(arm='panda_right'))))

    def test_diversity_never_backfills_duplicates_or_failed_poses(self):
        p=self.policy();p.proposed={}
        obj=NS(position_m=(-.2,0.,.8),size_m=(.1,.055,.045))
        choices=[self.candidate(),self.candidate(offset=.001),self.candidate(angle=math.pi)]
        self.assertEqual(len(p.diverse(choices,obj)),2)
        self.assertEqual(p.diverse(choices,obj),[])
        p.proposed={};p.failed_grasps['-0.2']=[vars(choices[0])]
        self.assertEqual(p.diverse(choices,obj),[choices[2]])
        self.assertEqual(p.diversity[-1]['similar_to_failed'],2)

    def test_failure_memory_changes_box_priority_without_banning_other_grasps(self):
        p=self.policy();p.visibility=lambda pair:dict(risk=.2)
        left=self.candidate();right=self.candidate(.2,arm='panda_right')
        report=dict(pair=dict(selection=[vars(left),vars(right)]),failure_reason='STATIC_PATH_BLOCKED:x',
                    path_failure=dict(arm='panda_right'))
        p.record_failure(report)
        failed=[NS(candidate=left),NS(candidate=right)]
        changed=[NS(candidate=left),NS(candidate=self.candidate(0.,arm='panda_right'))]
        self.assertLess(p.priority(changed),p.priority(failed))
        self.assertEqual(p.failed_grasps['-0.2'],[])
        self.assertEqual(len(p.failed_grasps['0.2']),1)
        # Source identity is geometry-based even when a new image assigns a new track ID.
        right.target_id='new-track';self.assertEqual(p.key(right),'0.2')

    def test_only_proven_empty_pre_descent_failures_can_recover(self):
        r=dict(last_phase='approach',motion=dict(ticks=20),failure_reason='STATIC_PATH_BLOCKED:fragment')
        self.assertTrue(recoverable_empty_failure(r))
        for change in (dict(empty_approach_started=True),dict(closure_started=True),
                       dict(last_phase='lift'),dict(failure_reason='JOINT_TRACKING_ERROR'),dict(motion=None)):
            self.assertFalse(recoverable_empty_failure(dict(r,**change)))

    def test_reference_failure_can_reobserve_only_before_descent_and_closure(self):
        r=dict(last_phase='pregrasp_recheck',motion=dict(ticks=939),
            failure_reason='PAIR_TARGET_UNSUPPORTED:panda_left:PRECLOSE_REFERENCE_EXPIRED')
        self.assertTrue(recoverable_empty_failure(r))
        for change in (dict(empty_approach_started=True),dict(closure_started=True),
                       dict(last_phase='close'),dict(last_phase='approach'),dict(motion=None),
                       dict(failure_reason='JOINT_TRACKING_ERROR')):
            self.assertFalse(recoverable_empty_failure(dict(r,**change)))
        self.assertTrue(recoverable_empty_failure(dict(r,failure_reason='PAIR_PREGRASP_REACQUISITION_LOST:panda_left')))

    def test_perception_failure_blacklists_culprit_pose_without_banning_other_arm(self):
        p=self.policy();left=self.candidate();right=self.candidate(.2,arm='panda_right')
        r=dict(pair=dict(selection=[vars(left),vars(right)]),
            failure_reason='PAIR_TARGET_UNSUPPORTED:panda_left:PRECLOSE_REFERENCE_EXPIRED',
            perception_failure=dict(arm='panda_left'))
        evidence=p.record_failure(r)
        self.assertEqual(evidence['culprit_arm'],'panda_left')
        self.assertEqual(len(p.failed_grasps['-0.2']),1)
        self.assertEqual(p.failed_grasps['0.2'],[])
        p.proposed={}
        self.assertEqual(p.diverse([left],NS(position_m=(-.2,0,.8),size_m=(.1,.055,.045))),[])

    def test_visibility_ranking_uses_frustum_and_occluding_spheres(self):
        camera=np.eye(4);K=np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
        points=np.array([[0.,0.,1.],[.05,0.,1.]])
        self.assertEqual(visible_fraction(points,camera,K,(100,100),[]),1.)
        self.assertEqual(visible_fraction(points,camera,K,(100,100),[(0,np.array([0.,0.,.5]),.1)]),0.)
        self.assertEqual(visible_fraction(points+[2,0,0],camera,K,(100,100),[]),0.)

    def test_more_than_two_sources_receive_arm_preferences(self):
        from workstation.diagnostics.graspgen_pair import preferred_assignments
        objects=[NS(track_id=str(i),position_m=(x,0.,.8)) for i,x in enumerate((-.2,-.1,.1,.2))]
        bases={a:dict(base_xyz=(x,.2,.8)) for a,x in [('panda_left',-.36),('panda_right',.36)]}
        self.assertEqual(list(preferred_assignments(objects,bases).values()),
                         ['panda_left','panda_left','panda_right','panda_right'])

    def test_joint_pair_selection_can_choose_other_boxes_before_motion(self):
        from workstation.diagnostics.graspgen_pair import paired_pregrasp
        from workstation.planning.selection import GraspCandidate,CandidateAssessment
        arms=('panda_left','panda_right');rows=[]
        for arm,names,x in [(arms[0],('a','c'),-.2),(arms[1],('b','d'),.2)]:
            for i,name in enumerate(names):
                c=GraspCandidate(name,arm,(x,0,.8),0,.07,.005,source='graspgen',candidate_id=name)
                rows.append(CandidateAssessment(c,True,(),dict(full_pose_checked=True,
                    pregrasp_joint_positions_rad=[.1]*7,slot_xy=(x,-.4)),10-i))
        robot=NS(checked_joint_plan=lambda a,s,g,scene,t:([s,g],1.),
                 spheres=lambda a,q:[(0,np.array([0.,float(a==arms[0]),1.]),.02)])
        states={a:NS(joint_positions_rad=np.zeros(7)) for a in arms}
        before,_=paired_pregrasp(robot,rows,states,None)
        self.assertEqual([c['target_id'] for c in before['selection']],['a','b'])
        policy=NS(priority=lambda p:(0,sum(r.candidate.target_id in ('a','b') for r in p)),
                  key=lambda c:c.target_id,visibility=lambda p:dict(risk=0.))
        after,_=paired_pregrasp(robot,rows,states,None,policy)
        self.assertEqual([c['target_id'] for c in after['selection']],['c','d'])
        self.assertEqual(after['joint_choice']['eligible_source_count'],4)
