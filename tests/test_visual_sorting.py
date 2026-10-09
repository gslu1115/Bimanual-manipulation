import unittest
from types import SimpleNamespace
import numpy as np
from workstation.skills.visual_sorting import observed_goal_assignment,VisualSortingTask
from workstation.perception.scene_estimate import ScenePriors,SceneEstimate


class SortingTests(unittest.TestCase):
    def obj(self,track='v1',x=0.,y=-.4,yaw=0.,failures=()):
        return SimpleNamespace(track_id=track,position_m=(x,y,.7825),size_m=(.1,.055,.045),
            posture='UPRIGHT',velocity_m_s=(0.,0.,0.),uncertainty_m=.003,axis_yaw_rad=yaw,failures=failures)

    def test_already_sorted_target_is_identified_without_truth_id(self):
        scene=SimpleNamespace(objects=[self.obj(),self.obj('v2',y=-.06)])
        self.assertEqual(observed_goal_assignment(scene,[(0,-.4)],.76),{0:'v1'})

    def test_ambiguous_or_unobserved_slot_is_not_counted(self):
        scene=SimpleNamespace(objects=[self.obj(),self.obj('v2')])
        self.assertEqual(observed_goal_assignment(scene,[(0,-.4)],.76),{})
        obj=self.obj();obj.velocity_m_s=None
        self.assertEqual(observed_goal_assignment(SimpleNamespace(objects=[obj]),[(0,-.4)],.76),{})

    def test_side_or_wrong_yaw_does_not_complete_sorting(self):
        obj=self.obj();obj.posture='SIDE'
        self.assertFalse(observed_goal_assignment(SimpleNamespace(objects=[obj]),[(0,-.4)],.76))
        self.assertFalse(observed_goal_assignment(SimpleNamespace(objects=[self.obj(yaw=.5)]),[(0,-.4)],.76))

    def test_uncertain_load_stops_task_without_another_skill(self):
        calls=[]
        class Fake:
            scene=None
            def __init__(self,*args,**kwargs):calls.append(1)
            def run(self):return {'success':False,'held_on_exit':True}
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        r=VisualSortingTask(None,None,priors,[(0,-.4)],max_items=1,skill_factory=Fake).run()
        self.assertEqual(r['stop_reason'],'UNCERTAIN_LOAD_SAFE_HOLD');self.assertEqual(len(calls),1)

    def test_no_executable_skill_stops_bounded_task(self):
        class Fake:
            scene=None
            def __init__(self,*args,**kwargs):pass
            def run(self):return {'success':False,'held_on_exit':False,'failure_reason':'NO_FEASIBLE_DIRECT_GRASP'}
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        r=VisualSortingTask(None,None,priors,[(0,-.4)],max_items=1,skill_factory=Fake).run()
        self.assertEqual(r['attempted_actions'],1);self.assertFalse(r['success'])

    def test_multiple_actions_share_tracker_and_skip_verified_slot_targets(self):
        test=self;calls=[]
        class Fake:
            def __init__(self,*args,**kwargs):
                calls.append(kwargs);n=len(calls)
                objects=[test.obj('v1',x=-.2)]
                if n==2:objects.append(test.obj('v2',x=.2))
                self.scene=SimpleNamespace(objects=objects,to_dict=lambda:{'objects':'fixture'})
                test.assertFalse(kwargs['eligible'](objects[0],self.scene))
            def run(self):return {'success':True,'held_on_exit':False}
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        result=VisualSortingTask(None,None,priors,[(-.2,-.4),(.2,-.4)],max_items=2,skill_factory=Fake).run()
        self.assertTrue(result['success']);self.assertEqual(result['attempted_actions'],2)
        self.assertIs(calls[0]['estimator'],calls[1]['estimator'])

    def test_failure_history_and_action_limit_are_retained(self):
        class Fake:
            scene=None
            def __init__(self,*args,**kwargs):pass
            def run(self):return {'success':False,'held_on_exit':False,'decision':{'candidate':{'target_id':'visual_v1'}}}
        priors=ScenePriors((.1,.055,.045),.76,(-.38,.38,-.51,.16))
        r=VisualSortingTask(None,None,priors,[(0,-.4)],max_items=1,max_actions=3,skill_factory=Fake).run()
        self.assertEqual(r['failure_history']['visual_v1'],3)
        self.assertEqual(r['stop_reason'],'ACTION_LIMIT_REACHED')
