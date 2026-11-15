import unittest
import numpy as np
from types import SimpleNamespace
from workstation.planning.plan_reuse import PreviewPlans,array_key,complete_ranked_pairs


class PreviewReuseTests(unittest.TestCase):
    def test_changed_start_goal_payload_or_observation_replans_without_using_old_path(self):
        plans=PreviewPlans()
        context=(3,4.,array_key(np.zeros(7)),array_key([.1,.2,.3]),array_key([.01,0,0]))
        original=[np.zeros((2,7)),np.ones((2,7))]
        plans.put('transfer',context,original,{'mode':'parallel'})
        paths,schedule=plans.get('transfer',context)
        self.assertEqual(schedule,{'mode':'parallel'})
        paths[0][:]=9
        self.assertFalse(plans.get('transfer',context)[0][0].any())
        for i in range(len(context)):
            changed=list(context);changed[i]=('changed',)
            self.assertIsNone(plans.get('transfer',tuple(changed)))
        self.assertIsNone(plans.get('lower',context))

    def test_multiple_previews_of_same_phase_do_not_replace_staging_plan(self):
        plans=PreviewPlans();path=[np.zeros((2,7)),np.ones((2,7))]
        plans.put('transfer',('stage',),path,{'stage':True})
        plans.put('transfer',('first_arm',),path,{'first':True})
        self.assertEqual(plans.get('transfer',('stage',))[1],{'stage':True})
        self.assertEqual(plans.get('transfer',('first_arm',))[1],{'first':True})

    def test_lazy_refinement_matches_exhaustive_ranking_and_skips_inferior_suffixes(self):
        estimates=[10,9,8,7,1];actual=[2,None,6,5,1]
        rows=[SimpleNamespace(id=i,score=s,feasible=True) for i,s in enumerate(estimates)]
        fixed=SimpleNamespace(id=99,score=0,feasible=True)
        pairs=[(r,fixed) for r in rows];calls=[];cache={}
        def complete(row):
            if row.id==99:return row
            if row.id not in cache:
                calls.append(row.id);s=actual[row.id]
                cache[row.id]=SimpleNamespace(id=row.id,score=s or 0,feasible=s is not None)
            return cache[row.id]
        rank=lambda pair:(-sum(r.score for r in pair),)
        best=next(complete_ranked_pairs(pairs,rank,complete))
        self.assertEqual(best[0].id,2)
        self.assertEqual(calls,[0,1,2,3])
        self.assertNotIn(4,calls)
        # Iterating all survivors has the exact exhaustive result, including
        # a high optimistic candidate demoted by its measured path clearance.
        ordered=list(complete_ranked_pairs(pairs,rank,complete))
        expected=sorted((i for i,s in enumerate(actual) if s is not None),key=lambda i:-actual[i])
        self.assertEqual([pair[0].id for pair in ordered],expected)
