import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from workstation.planning.selection import CandidateAssessment,GraspCandidate
from workstation.diagnostics.graspgen_pair import paired_pregrasp


class RankedPickParity(unittest.TestCase):
    def test_tied_resampled_candidate_follows_previous_batch(self):
        from workstation.planning.plan_reuse import refine_pick_pairs
        def row(arm,identifier,group):
            c=GraspCandidate('box',arm,(0,0,0),0,.07,0,candidate_id=identifier)
            return CandidateAssessment(c,True,(),{'source_order':(group,0)},1.)
        first=row('left','old',8)
        backup=row('left','new',16)
        right=row('right','right',4)
        pairs=[(backup,right),(first,right)]
        ordered=list(refine_pick_pairs(pairs,lambda p:(-sum(r.score for r in p),),lambda r:r,lambda r:r))
        self.assertEqual(ordered,[(first,right),(backup,right)])

    def test_same_inputs_ties_rejections_and_crossing_paths(self):
        rng=np.random.default_rng(7731)
        arms=('panda_left','panda_right')
        states={arm:SimpleNamespace(joint_positions_rad=np.zeros(7)) for arm in arms}
        policy=SimpleNamespace(priority=lambda pair:(0,0.),key=lambda c:c.target_id,visibility=lambda pair:{})
        def spheres(arm,q):
            sign=1 if arm==arms[0] else -1
            return [(5,np.array([sign*(.55-q[0]),0.,0.]),.06)]
        robot=SimpleNamespace(checked_joint_plan=lambda arm,start,end,*_:([start,end],1.),spheres=spheres)
        for trial in range(70):
            bounds=[];prefixes={};finals={}
            for slot in range(2):
                rows=[]
                for i in range(8):
                    arm=arms[i%2];box=str(i//2)
                    c=GraspCandidate(box,arm,(0,0,0),0,.07,0,source='graspgen',candidate_id=str(i))
                    key=i,slot
                    upper=int(rng.integers(1,5));prefix=upper-int(rng.integers(0,3))
                    terms=dict(source_order=(slot,i),slot_xy=(slot,0),ranking_bound=True,
                               pregrasp_joint_positions_rad=[(.8 if trial==0 else .1*(i+1))]*7)
                    bound=CandidateAssessment(c,True,(),terms,upper)
                    front=replace(bound,feasible=bool(rng.random()>.15),score=prefix,
                                  terms=dict(terms,ranking_bound=False,full_pose_checked=True))
                    prefixes[key]=front
                    finals[key]=replace(front,feasible=front.feasible and rng.random()>.15,
                                        score=prefix-int(rng.integers(0,3)))
                    rows.append(bound)
                bounds.extend(sorted(rows,key=lambda r:(-r.score,r.candidate.target_id,r.candidate.arm)))
            def key(r):return int(r.candidate.candidate_id),r.terms['slot_xy'][0]
            original=sorted(prefixes.values(),key=lambda r:(r.terms['source_order'][0],not r.feasible,-r.score,r.candidate.target_id,r.candidate.arm,r.terms['source_order'][1]))
            finish=lambda r:finals[key(r)]
            a,old_rejections=paired_pregrasp(robot,original,states,None,policy,finish)
            b,new_rejections=paired_pregrasp(robot,bounds,states,None,policy,finish,prefix_assessment=lambda r:prefixes[key(r)])
            self.assertEqual(old_rejections,new_rejections)
            for field in ('selection','slots','pregrasp_knots','min_clearance_m','sample_count'):
                self.assertEqual(a and a[field],b and b[field],(trial,field))

