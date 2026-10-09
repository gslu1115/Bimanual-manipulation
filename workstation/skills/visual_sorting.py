"""Bounded observed-target task loop; privileged IDs/counts never enter decisions."""
from dataclasses import asdict
import math
import numpy as np
from workstation.perception.rgbd_cartons import CartonEstimator, axis_error
from workstation.skills.visual_pick_place import VisualPickPlace


def observed_goal_assignment(scene, slots, support_z):
    """One visual object per slot, with pose and observed-motion evidence."""
    assigned={}
    for obj in scene.objects:
        if obj.failures or obj.posture!='UPRIGHT' or obj.velocity_m_s is None:continue
        if obj.uncertainty_m>.008 or np.linalg.norm(obj.velocity_m_s)>.015:continue
        if abs(obj.position_m[2]-support_z-obj.size_m[2]/2)>.008:continue
        if axis_error(obj.axis_yaw_rad,0)>math.radians(8):continue
        nearest=min(range(len(slots)),key=lambda i:np.linalg.norm(np.asarray(obj.position_m[:2])-slots[i]))
        if np.linalg.norm(np.asarray(obj.position_m[:2])-slots[nearest])>.018:continue
        if nearest in assigned:
            # An ambiguous duplicate cannot verify an occupied target slot.
            assigned[nearest]=None
        else:assigned[nearest]=obj.track_id
    return {index:track for index,track in assigned.items() if track is not None}


class VisualSortingTask:
    def __init__(self,robot,read_packet,priors,slots,save_scene=None,trace_scene=None,
                 segmenter=None,max_items=3,max_actions=6,skill_factory=VisualPickPlace):
        self.robot,self.read_packet,self.priors,self.slots=robot,read_packet,priors,slots
        self.save_scene=save_scene or (lambda *args:None)
        self.trace_scene=trace_scene
        self.segmenter=segmenter;self.estimator=CartonEstimator(priors,segmenter)
        self.max_items,self.max_actions=int(max_items),int(max_actions)
        if not 1<=self.max_items<=len(slots) or not 1<=self.max_actions<=20:
            raise ValueError('Bounded task requires 1..slot_count items and 1..20 actions')
        self.factory=skill_factory;self.failures={}

    def run(self):
        result=dict(success=False,scope='bounded_observed_cartons_not_entire_unknown_workspace',
                    requested_items=self.max_items,max_actions=self.max_actions,
                    actions=[],failure_history=self.failures,box_attachment=False,box_pose_teleport=False)
        for action in range(self.max_actions):
            def eligible(obj,scene):
                assigned=observed_goal_assignment(scene,self.slots,self.priors.support_z_m)
                return obj.track_id not in assigned.values()
            def save(label,packet,scene):
                self.save_scene('item_'+str(action+1).zfill(2)+'_'+label,packet,scene)
            skill=self.factory(self.robot,self.read_packet,self.priors,self.slots,save,self.trace_scene,
                self.segmenter,estimator=self.estimator,failure_history=self.failures,eligible=eligible)
            child=skill.run();result['actions'].append(child)
            if skill.scene is not None:
                assigned=observed_goal_assignment(skill.scene,self.slots,self.priors.support_z_m)
                result['observed_goal_slots']={str(k):v for k,v in assigned.items()}
                result['final_scene']=skill.scene.to_dict()
                if child['success'] and len(assigned)>=self.max_items:
                    result['success']=True;result['stop_reason']='REQUESTED_OBSERVED_ITEMS_VERIFIED';break
            if child.get('held_on_exit'):
                result['stop_reason']='UNCERTAIN_LOAD_SAFE_HOLD';break
            if not child['success']:
                candidate=child.get('decision',{}).get('candidate',{})
                if candidate:
                    track=candidate['target_id'];self.failures[track]=self.failures.get(track,0)+1
                else:
                    result['stop_reason']=child.get('failure_reason','NO_EXECUTABLE_SKILL');break
        else:result['stop_reason']='ACTION_LIMIT_REACHED'
        result['attempted_actions']=len(result['actions'])
        return result
