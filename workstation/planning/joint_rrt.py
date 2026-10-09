"""Bounded joint-space RRT-Connect with caller-supplied full edge checks.

Search exhaustion is not proof that a target is unreachable. This planner does
not relax collisions, unknown space, joint limits, or the caller's sample grid.
"""
import time
import numpy as np


def bounded_joint_rrt(start,goal,limits,edge_valid,seed=0,iterations=160,seconds=15.,step=.12):
    start,goal=np.asarray(start,float),np.asarray(goal,float)
    lower,upper=np.asarray(limits[0]),np.asarray(limits[1])
    if (start.ndim!=1 or goal.shape!=start.shape or lower.shape!=start.shape or upper.shape!=start.shape
            or not all(np.isfinite(a).all() for a in (start,goal,lower,upper)) or np.any(lower>=upper)):
        return None,dict(reason='INVALID_JOINT_INPUT',iterations=0)
    if np.any(start<lower) or np.any(start>upper) or np.any(goal<lower) or np.any(goal>upper):
        return None,dict(reason='JOINT_LIMIT',iterations=0)
    if not edge_valid(goal,goal):return None,dict(reason='GOAL_STATE_NOT_VERIFIED',iterations=0)
    if not edge_valid(start,start):return None,dict(reason='START_STATE_NOT_VERIFIED',iterations=0)
    rng=np.random.default_rng(seed);began=time.monotonic();trees=[([start],[-1]),([goal],[-1])]
    searched=0;checks=0
    lo=np.maximum(lower,np.minimum(start,goal)-.5);hi=np.minimum(upper,np.maximum(start,goal)+.5)
    def grow(tree,target):
        nonlocal checks
        nodes,parents=tree
        nearest=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)))
        origin=nodes[nearest];delta=target-origin
        node=origin+delta*min(1.,step/max(1e-12,float(np.max(np.abs(delta)))))
        checks+=1
        if not edge_valid(origin,node):return None
        nodes.append(node);parents.append(nearest)
        return len(nodes)-1
    def chain(tree,index):
        nodes,parents=tree;out=[]
        while index>=0:out.append(nodes[index]);index=parents[index]
        return out[::-1]
    for searched in range(1,iterations+1):
        if time.monotonic()-began>seconds:break
        side=(searched-1)%2;a,b=trees[side],trees[1-side]
        target=b[0][0] if rng.random()<.25 else rng.uniform(lo,hi)
        ai=grow(a,target)
        if ai is None:continue
        for _ in range(30):
            if time.monotonic()-began>seconds:break
            bi=grow(b,a[0][ai])
            if bi is None:break
            if np.max(np.abs(b[0][bi]-a[0][ai]))<1e-8:
                left,right=(chain(a,ai),chain(b,bi)) if side==0 else (chain(b,bi),chain(a,ai))
                path=left+right[::-1][1:]
                # Independently recheck the returned path, not just tree growth.
                if all(edge_valid(q0,q1) for q0,q1 in zip(path,path[1:])):
                    return path,dict(reason=None,iterations=searched,edge_checks=checks,
                                     elapsed_s=time.monotonic()-began,search_scope='bounded local c-space')
    return None,dict(reason='BOUNDED_SEARCH_EXHAUSTED',iterations=searched,edge_checks=checks,
                     elapsed_s=time.monotonic()-began,search_scope='bounded local c-space',
                     unreachable_proven=False)
