"""Visual pair selection and conveyor interlock; no simulator object state."""
import math
import numpy as np
from workstation.perception.scene_estimate import pose_is_observed


def match_stationary_sources(scene,references):
    matched={}
    for key,ref in references.items():
        candidates=[o for o in scene.objects if not o.failures and o.posture=='UPRIGHT'
            and abs(o.time_s-scene.observation_time_s)<=1e-6
            and pose_is_observed(o,scene) and np.linalg.norm(np.asarray(o.position_m)-ref['position_m'])<=.005
            and ('axis_yaw_rad' not in ref or
                 abs((o.axis_yaw_rad-ref['axis_yaw_rad']+math.pi/2)%math.pi-math.pi/2)<=math.radians(3))
            and np.allclose(o.size_m,ref['size_m'],atol=.002,rtol=0)]
        if len(candidates)==1:matched[key]=candidates[0]
    if len({o.track_id for o in matched.values()})!=len(matched):return {}
    return matched


def match_task_sources(scene,references):
    """Only candidates for the NEXT pick need a current usable pose.

    Persistent identity tolerates measured motion. It never supplies an old pose
    for an invisible source, and missing unrelated sources do not veto this pick.
    """
    result={}
    for obj in scene.objects:
        key=getattr(obj,'identity_id',None)
        if (key in references and not obj.failures and obj.posture=='UPRIGHT' and
                pose_is_observed(obj,scene) and abs(obj.time_s-scene.observation_time_s)<=1e-6 and
                0<=scene.assembled_time_s-obj.time_s<=.15 and
                np.allclose(obj.size_m,references[key]['size_m'],atol=1e-6,rtol=0)):
            if key in result:return {}
            result[key]=obj
    return result


def choose_source_pair(objects,bases):
    """Choose the currently observed front carton in each arm's half-workspace."""
    mid=sum(v['base_xyz'][0] for v in bases.values())/len(bases)
    left=[o for o in objects if o.position_m[0]<mid]
    right=[o for o in objects if o.position_m[0]>=mid]
    if not left or not right:raise RuntimeError('PAIR_SOURCE_ALLOCATION_UNAVAILABLE')
    return [min(side,key=lambda o:(o.position_m[1],o.track_id)) for side in (left,right)]


class VisualConveyorIndex:
    """Only fresh depth free-space and measured empty-arm clearance finish indexing."""
    visual_control=True

    def __init__(self,config):
        c=config['conveyor'];self.speed=c['speed'];self.minimum_s=c['index_distance']/self.speed
        self.timeout_s=c['timeout_s'];self.positions=tuple(tuple(xy) for xy in c['slots_xy'])
        self.state='WAITING';self.completed_batches=0
        self.occupied=[None]*len(c['slots_xy']);self.delivered=set();self.events=[]
        self.elapsed=0.;self.distance=0.;self.stable=None;self.batch=[];self.last_observation=None

    def start(self,verified_ids,arms_clear):
        if (self.state!='WAITING' or not arms_clear or len(verified_ids)!=2 or len(set(verified_ids))!=2
                or self.delivered.intersection(verified_ids)):
            raise RuntimeError('VISUAL_INDEX_START_UNSUPPORTED')
        self.batch=list(verified_ids);self.state='INDEXING';self.elapsed=0.;self.distance=0.
        self.stable=None;self.last_observation=None
        self.events.append(dict(event='index_started',visual_placements=self.batch))

    def advance(self,dt):
        if self.state!='INDEXING':return 0.
        self.elapsed+=dt;self.distance=self.elapsed*self.speed
        if self.elapsed>self.timeout_s:
            self.state='FAULT';raise RuntimeError('VISUAL_INDEX_TIMEOUT')
        return self.speed

    def observe(self,time_s,views,arms_clear):
        if self.state!='INDEXING':raise RuntimeError('VISUAL_INDEX_NOT_ACTIVE')
        if not arms_clear:
            self.state='FAULT';raise RuntimeError('VISUAL_INDEX_ARM_INTERLOCK')
        if self.last_observation is not None and time_s<=self.last_observation:
            raise RuntimeError('VISUAL_INDEX_STALE_OBSERVATION')
        self.last_observation=time_s
        clear=(len(views)==len(self.occupied) and {tuple(v['xy']) for v in views}==set(self.positions)
               and all(v['free'] and not v['observed_objects'] and
               abs(v['observation_time_s']-time_s)<1e-6 for v in views))
        if not clear or self.elapsed<self.minimum_s:self.stable=None
        elif self.stable is None:self.stable=time_s
        elif time_s-self.stable>=.5:
            self.state='WAITING';self.completed_batches+=1;self.delivered.update(self.batch)
            self.events.append(dict(event='visually_cleared',time_s=time_s,elapsed_s=self.elapsed,
                                    commanded_distance_m=self.distance,slot_views=views))
            return True
        return False
