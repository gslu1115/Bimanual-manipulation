"""Reuse calculations within unchanged sensor/encoder contexts, never old evidence."""
import numpy as np


def array_key(value):
    a=np.asarray(value,dtype=np.float64)
    return a.shape,a.tobytes()


def encoder_key(states):
    return tuple((arm,array_key(s.joint_positions_rad),array_key(s.finger_positions_m))
                 for arm,s in sorted(states.items()))


class CandidatePrefixCache:
    """One live snapshot's slot-independent pick checks; never serialized."""
    def __init__(self):
        self.pregrasps={}
        self.destinations={}
        self.path_hits=0
        self.context=None;self.rows={};self.hits=0;self.misses=0

    def key(self,robot,candidate,obj,scene,hint,include_lift):
        context=(scene,getattr(robot,'observed_workspace',None),
                 getattr(robot,'stationary_payload_keepout',None))
        states=encoder_key(robot.read_states())
        # Identity comparisons avoid ndarray equality and retain the source
        # objects, preventing id reuse. A new frame/scene always invalidates.
        if (self.context is None or any(a is not b for a,b in zip(context,self.context[:3]))
                or states!=self.context[3]):
            self.rows.clear();self.pregrasps.clear();self.destinations.clear();self.context=(*context,states)
        return id(candidate),id(obj),None if hint is None else array_key(hint),include_lift

    def get(self,key):
        value=self.rows.get(key)
        if value is None:self.misses+=1
        else:self.hits+=1
        return None if value is None else value[2]

    def put(self,key,candidate,obj,value):
        self.rows[key]=(candidate,obj,value)

    def report(self):
        return dict(hits=self.hits,misses=self.misses,entries=len(self.rows),
                    scope='same live scene, workspace, encoders and candidate; no persisted IK')


class PreviewPlans:
    """Exact-context path reuse; a changed context takes the normal planner."""
    def __init__(self):
        self.rows={};self.hits=0;self.misses=0

    def put(self,phase,context,paths,schedule):
        self.rows[phase,context]=([np.array(p,copy=True) for p in paths],schedule)

    def get(self,phase,context):
        row=self.rows.get((phase,context))
        if row is None:
            self.misses+=1
            return None
        self.hits+=1
        return [p.copy() for p in row[0]],row[1]


def _assess_pair(pair, assess, rejected, skipped, stage):
    # Each input row describes one candidate and slot in this snapshot.
    # Retain the actual rejected row so its identity stays valid for this search.
    if any(id(row) in rejected for row in pair):
        if skipped is not None:skipped[stage]+=1
        return None
    left,right=pair
    checked_left=assess(left)
    if not checked_left.feasible:
        rejected[id(left)]=left
        if skipped is not None:skipped[stage]+=1
        return None
    checked_right=assess(right)
    if not checked_right.feasible:
        rejected[id(right)]=right
        return None
    return checked_left,checked_right


def complete_ranked_pairs(pairs,rank,complete,skipped=None):
    """Best-first exact refinement of optimistic scores, not a new policy.

    Prefix clearance upper-bounds full-path clearance. The remaining rank
    terms do not change: visibility uses the same checked pregrasp joints.
    Refine and reinsert before yielding, so no unchecked optimistic candidate
    can displace a fully checked better pair. Failed suffixes remain rejected.
    """
    import heapq
    rejected={}
    pending=[(rank(pair),i,pair) for i,pair in enumerate(pairs)]
    heapq.heapify(pending)
    while pending:
        optimistic,index,pair=heapq.heappop(pending)
        checked=_assess_pair(pair,complete,rejected,skipped,'destination')
        if checked is None:continue
        actual=rank(checked)
        if actual!=optimistic:
            heapq.heappush(pending,(actual,index,checked))
            continue
        yield checked

def refine_pick_pairs(pairs,rank,prefix,complete,skipped=None):
    """Refine upper bounds in order, including the original prefix tie order."""
    import heapq
    def row_order(row):
        slot,proposal=row.terms['source_order']
        return slot,-row.score,row.candidate.target_id,row.candidate.arm,proposal

    def item(current,front,stage,index):
        return (rank(current),rank(front),tuple(map(row_order,front)),index,stage,current,front)

    queue=[item(pair,pair,0,i) for i,pair in enumerate(pairs)]
    heapq.heapify(queue)
    rejected=({}, {})
    while queue:
        _,_,_,index,stage,pair,front=heapq.heappop(queue)
        if stage==2:
            yield pair
            continue
        callback=prefix if stage==0 else complete
        checked=_assess_pair(pair,callback,rejected[stage],skipped,('prefix','destination')[stage])
        if checked is not None:
            heapq.heappush(queue,item(checked,checked if stage==0 else front,stage+1,index))
